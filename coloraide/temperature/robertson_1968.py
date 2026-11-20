"""
Calculate CCT with Robertson 1968 method.

Uses Robertson 1968 method.

- https://en.wikipedia.org/wiki/Correlated_color_temperature#Robertson's_method
- http://www.brucelindbloom.com/index.html?Math.html
"""
from __future__ import annotations
import math
from . import planck
from .. import algebra as alg
from .. import util
from .. import cat
from .. import cmfs
from . import CCT
from ..types import Vector, VectorLike
from typing import Any, TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:  #pragma: no cover
    from ..color import Color

# Original 31 mired points 0 - 600
MIRED_ORIGINAL = tuple(range(0, 100, 10)) + tuple(range(100, 601, 25))
# Extended 16 mired points 625 - 1000
MIRED_EXTENDED = MIRED_ORIGINAL + tuple(range(625, 1001, 25))


class CCTEntry(NamedTuple):
    """CCT LUT entry."""

    mired: float
    u: float
    v: float
    slope: float
    slope_length: float
    du: float
    dv: float


class Robertson1968(CCT):
    """Delta E plugin class."""

    NAME = 'robertson-1968'
    CHROMATICITY = 'uv-1960'

    def __init__(
        self,
        cmfs: cmfs.CMFs = cmfs.CIE_1931_2DEG,
        white: VectorLike = cat.WHITES['2deg']['D65'],
        mired: VectorLike = MIRED_EXTENDED,
        sigfig: int = 5,
        planck_step: int = 1,
    ) -> None:
        """Initialize."""

        self.white = white
        self.table = self.generate_table(cmfs, white, mired, sigfig, planck_step)

    def generate_table(
        self,
        cmfs: cmfs.CMFs,
        white: VectorLike,
        mired: VectorLike,
        sigfig: int,
        planck_step: int,
    ) -> list[CCTEntry]:
        """
        Generate the necessary table for the Robertson1968 method.

        The below algorithm, coupled with the 1nm CMFs for the 1931 2 degree table, allows us to replicate the
        documented 31 points exactly.

        For each mired value we calculate two additional points, one on each side at a distance of 0.1.
        We use a very small distance so that we can approximate the slope. We calculate the distance between
        the targeted value and the two neighbors and then calculate the slope of the two small lines. Then we
        can calculate an interpolation factor and interpolate the slope for our target.

        We are able to calculate the uv pair for each mired point directly except for 0. 0 requires us to
        interpolate the values as it will cause a divide by zero in the Planckian locus. In this case, we
        assume a perfect 0.5 (middle) for our interpolation.

        Additionally, we precalculate a few other things to save time:
        - slope length of unit vector
        - u component of slope unit vector
        - v component of slope unit vector
        """

        xyzw = util.xy_to_xyz(white)
        table = []  # type: list[CCTEntry]
        to_uv = util.xy_to_uv_1960 if self.CHROMATICITY == 'uv-1960' else util.xy_to_uv
        for t in mired:
            uv1 = to_uv(planck.temp_to_xy_planckian_locus(1e6 / (t - 0.01), cmfs, xyzw, step=planck_step))
            uv2 = to_uv(planck.temp_to_xy_planckian_locus(1e6 / (t + 0.01), cmfs, xyzw, step=planck_step))
            if t == 0:
                factor = 0.5
                uv = [alg.lerp(uv1[0], uv2[0], factor), alg.lerp(uv1[1], uv2[1], factor)]
            else:
                uv = to_uv(planck.temp_to_xy_planckian_locus(1e6 / t, cmfs, xyzw, step=planck_step))
                d1 = math.sqrt((uv[1] - uv1[1]) ** 2 + (uv[0] - uv1[0]) ** 2)
                d2 = math.sqrt((uv2[1] - uv[1]) ** 2 + (uv2[0] - uv[0]) ** 2)
                factor = d1 / (d1 + d2)

            # Attempt to calculate the slope, if it falls exactly where the slope switch,
            # There will be a divide by zero, just skip this location.
            try:
                m1 = -((uv[1] - uv1[1]) / (uv[0] - uv1[0])) ** -1
                m2 = -((uv2[1] - uv[1]) / (uv2[0] - uv[0])) ** -1
            except ZeroDivisionError:  # pragma: no cover
                continue

            m = alg.lerp(m1, m2, factor)
            if sigfig:
                template = f'{{:.{sigfig}g}}'
                slope = float(template.format(m))
                length = math.sqrt(1 + slope * slope)

                table.append(
                    CCTEntry(
                        float(t),
                        float(template.format(uv[0])),
                        float(template.format(uv[1])),
                        slope,
                        length,
                        1 / length,
                        slope / length
                    )
                )
            else:
                length = math.sqrt(1 + m * m)
                table.append(CCTEntry(t, uv[0], uv[1], m, length, 1 / length, m / length))
        return table

    def calc_du_dv(
        self,
        previous: CCTEntry,
        current: CCTEntry,
        factor: float
    ) -> tuple[float, float]:
        """
        Calculate the unit normal of the isotherm.

        Each entry stores one of the two isotherm normals, written as `(1, slope)` normalized,
        so the stored direction is by construction the normal that faces increasing `u`.
        Because the locus `u` increases strictly with mired, the tangent is never vertical and
        that normal's `v` component can never be zero: it is bounded away from zero (minimum
        |dv| = 0.2365 over the default table), which is what makes the fix below numerically safe.

        Which of the two normals `(1, slope)` actually is depends on the sign of the tangent's
        `v` component, and the locus `v` is not monotonic. It peaks inside the table's slope
        sign change interval (mired 600/625, i.e. 1600K - 1666.7K for D65), where `slope`
        crosses from negative to positive. So the stored direction is the "above the curve"
        normal on the low temperature side of that vertex and the "below the curve" normal on
        the high temperature side, and a stored or interpolated `dv` carries no reliable Duv
        sign. That is the source of the historical inversion, which only ever affected the
        low side of the vertex.

        The orientation is canonicalized here, and only here, to the documented convention:
        a positive Duv lies on the increasing `v` side ("above the curve"). Both `to_cct` and
        `from_cct` consume this same normal, so reading and generating agree, and a custom
        table cannot reintroduce the split because nothing outside this method assumes a
        particular stored orientation.
        """

        pslope = previous.slope
        slope = current.slope
        u1 = previous.du
        v1 = previous.dv
        u2 = current.du
        v2 = current.dv

        # Align the two isotherms when the pair straddles the locus vertex (slope sign change)
        if (pslope * slope) < 0:
            u2 *= -1
            v2 *= -1

        # Interpolate the orientation across the pair.
        du = alg.lerp(u2, u1, factor)
        dv = alg.lerp(v2, v1, factor)
        length = math.sqrt(du ** 2 + dv ** 2)
        du /= length
        dv /= length

        # Canonical orientation: positive Duv is always on the increasing `v` side.
        if dv < 0:
            du = -du
            dv = -dv

        return du, dv

    def to_cct(self, color: Color, **kwargs: Any) -> Vector:
        """Calculate a color's CCT."""

        dip = kelvin = duv = 0.0
        u, v = color.split_chromaticity(self.CHROMATICITY)[:-1]
        end = len(self.table) - 1

        # Search for the pair of isotherms that the coordinate falls between.
        for index, current in enumerate(self.table):
            # Signed distance from this isotherm. The sign convention is normalized by the
            # slope direction so that crossing the locus is always a change from positive to
            # negative, regardless of which side of the locus vertex the row is on.
            slope = current.slope
            if slope < 0:
                di = (v - current.v) - slope * (u - current.u)
            else:
                di = (current.v - v) - slope * (current.u - u)

            if index > 0 and (di <= 0.0 or index == end):
                # Calculate the required interpolation factor between the two lines
                previous = self.table[index - 1]
                di /= current.slope_length
                dip /= previous.slope_length
                factor = dip / (dip - di)

                # Calculate the temperature. If the mired value is zero, assume infinity.
                pmired = previous.mired
                mired = (pmired - factor * (pmired - current.mired))
                kelvin = 1.0E6 / mired if mired else math.inf

                # Calculate Duv. The normal is already canonicalized so that a positive
                # value lies on the increasing `v` side, so no further sign is applied.
                du, dv = self.calc_du_dv(previous, current, 1 - factor)
                duv = (
                    du * (u - alg.lerp(previous.u, current.u, factor)) +
                    dv * (v - alg.lerp(previous.v, current.v, factor))
                )

                break

            # Save distance as previous
            dip = di

        return [kelvin, duv]

    def from_cct(
        self,
        kelvin: float,
        duv: float,
        **kwargs: Any
    ) -> tuple[tuple[float, float], str]:
        """Calculate a color that satisfies the CCT."""

        # Find inverse temperature to use as index.
        mired = 1.0E6 / kelvin
        u = v = 0.0
        end = len(self.table) - 2

        for index, current in enumerate(self.table):
            future = self.table[index + 1]

            # Find the two isotherms that our target temp is between
            future_mired = future.mired
            if mired < future_mired or index == end:
                # Find relative weight between the two values
                f = (future_mired - mired) / (future_mired - current.mired)

                # Interpolate the uv coordinates of our target temperature
                u = alg.lerp(future.u, current.u, f)
                v = alg.lerp(future.v, current.v, f)

                # Calculate the offset along the slope. The normal is canonicalized
                # (positive Duv is on the increasing `v` side), so generating and reading
                # a color use the same orientation and round trip consistently.
                if duv:
                    du, dv = self.calc_du_dv(current, future, f)
                    u += du * duv
                    v += dv * duv
                break

        return (u, v), self.CHROMATICITY
