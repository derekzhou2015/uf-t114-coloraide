"""Test CCT."""
import math
from coloraide.everything import ColorAll as Color
from coloraide import cmfs
from . import util
import pytest
import unittest


class TestOhno2013Temp(util.ColorAssertsPyTest):
    """Test Ohno 2013."""

    @pytest.mark.parametrize('cct', list(range(1000, 100000, 100)))
    def test_cct(self, cct):
        """
        Test CCT methods.

        We test at a high resolution to make sure the spline implementation is working as good as without.
        """

        for duv in (-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03):
            cct2, duv2 = Color.blackbody('xyz-d65', cct, duv, scale=False, method='ohno-2013').cct(method='ohno-2013')
            assert math.isclose(cct2, cct, rel_tol=(0.00001 if cct < 84000 else 0.0001), abs_tol=0.000001)
            assert math.isclose(duv2, duv, rel_tol=0.000001, abs_tol=0.000001)


class TestRobertson1968(util.ColorAssertsPyTest):
    """Test Robertson 1968."""

    @pytest.mark.parametrize('cct', list(range(1667, 100000, 1000)))
    def test_cct(self, cct):
        """Test CCT methods."""

        for duv in (-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03):
            cct2, duv2 = Color.blackbody(
                'xyz-d65', cct, duv, scale=False, method='robertson-1968'
            ).cct(method='robertson-1968')
            assert math.isclose(cct2, cct, rel_tol=0.00001, abs_tol=0.00001)
            assert math.isclose(duv2, duv, rel_tol=0.00001, abs_tol=0.00001)


class TestOhno2013Color(util.ColorAssertsPyTest):
    """Test Ohno 2013."""

    @pytest.mark.parametrize(
        'color',
        [
            'green',
            'yellow',
            'orange',
            'white',
            'gray',
            'pink',

        ]
    )
    def test_cct(self, color):
        """Test CCT methods."""

        cct, duv = Color(color).cct()
        c = Color.blackbody('xyz-d65', cct, duv, scale=False, method='ohno-2013')
        cct2, duv2 = c.cct(method='ohno-2013')
        assert math.isclose(cct, cct2, rel_tol=0.00001, abs_tol=0.00001)
        assert math.isclose(duv, duv2, rel_tol=0.00001, abs_tol=0.00001)


class TestRobertson1968Color(util.ColorAssertsPyTest):
    """Test Robertson 1968."""

    @pytest.mark.parametrize(
        'color',
        [
            'green',
            'yellow',
            'orange',
            'white',
            'gray',
            'pink',

        ]
    )
    def test_cct(self, color):
        """Test CCT methods."""

        cct, duv = Color(color).cct(method='robertson-1968')
        c = Color.blackbody('xyz-d65', cct, duv, scale=False, method='robertson-1968')
        cct2, duv2 = c.cct(method='robertson-1968')
        assert math.isclose(cct, cct2, rel_tol=0.00001, abs_tol=0.00001)
        assert math.isclose(duv, duv2, rel_tol=0.00001, abs_tol=0.00001)


class TestCCTSpecificCases(util.ColorAsserts, unittest.TestCase):
    """Test CCT specific cases."""

    def test_bad_algorithm(self):
        """Test bad algorithm inputs."""

        with self.assertRaises(ValueError):
            Color.blackbody('srgb', 2000, method='bad')

        with self.assertRaises(ValueError):
            Color('blue').cct(method='bad')

    def test_outside_gamut(self):
        """Test temperature outside of gamut."""

        self.assertColorEqual(
            Color.blackbody('srgb-linear', 1500, scale_space='srgb-linear'),
            Color('color(srgb-linear 1 0.14966 0)')
        )

    def test_output_space(self):
        """Test output space."""

        self.assertColorEqual(
            Color.blackbody('display-p3', 2000, scale_space='srgb-linear'),
            Color('color(display-p3 0.93958 0.56696 0.2294)')
        )

    def test_normalization_space(self):
        """Test normalization space."""

        self.assertColorEqual(
            Color.blackbody('display-p3', 2000, scale_space='display-p3-linear'),
            Color('color(display-p3 1 0.60474 0.24668)')
        )

    def test_no_normalization_space(self):
        """Test normalization space."""

        self.assertColorEqual(
            Color.blackbody('xyz-d65', 2000, scale=False),
            Color('color(xyz-d65 1.2743 1 0.14517)')
        )

    def test_ohno_exact(self):
        """Test exact CMFs."""

        cct1, duv1 = Color('orange').cct(method='ohno-2013')
        assert math.isclose(cct1, 2424.1146637385255, rel_tol=1e-11, abs_tol=1e-11)
        assert math.isclose(duv1, 0.008069417642630583, rel_tol=1e-11, abs_tol=1e-11)
        cct2, duv2 = Color('orange').cct(method='ohno-2013', exact=True)
        assert math.isclose(cct2, 2424.1146637385255, rel_tol=1e-11, abs_tol=1e-11)
        assert math.isclose(duv2, 0.008069417642630583, rel_tol=1e-11, abs_tol=1e-11)

    def test_ohno_alternate_cmfs(self):
        """Test alternate CMFs."""

        from coloraide.temperature.ohno_2013 import Ohno2013
        from coloraide import cat

        class Custom(Color):
            CCT = 'ohno-2013'

        Custom.register(
            Ohno2013(cmfs.CIE_1931_2DEG, cat.WHITES['2deg']['D50']),
            overwrite=True
        )

        srgbl = Custom.blackbody('srgb-linear', 5000, scale_space='srgb-linear')
        self.assertColorEqual(srgbl, Color('color(srgb-linear 1 0.7915 0.6282)'))
        cct, duv = srgbl.cct()
        assert math.isclose(cct, 5000.0053320884645, rel_tol=1e-11, abs_tol=1e-11)
        assert math.isclose(duv, 2.942006042494333e-08, rel_tol=1e-11, abs_tol=1e-11)

    def test_robertson_custom_table(self):
        """Test that we can customize Robertson 1968 table."""

        from coloraide.temperature import robertson_1968
        from coloraide import cmfs
        from coloraide import cat

        class Custom(Color):
            CCT = 'robertson-1968'

        Custom.register(
            robertson_1968.Robertson1968(cmfs.CIE_1931_2DEG, cat.WHITES['2deg']['D50']),
            overwrite=True
        )

        self.assertEqual(Custom.blackbody('xyz-d65', 1000, scale=False).cct(), [1000.0, 0.0])

    def test_robertson_sigfig(self):
        """Test that significant figure option."""

        from coloraide.temperature import robertson_1968
        from coloraide import cmfs
        from coloraide import cat

        class Custom(Color):
            CCT = 'robertson-1968'

        # Don't use significant figure rounding
        Custom.register(
            robertson_1968.Robertson1968(cmfs.CIE_1931_2DEG, cat.WHITES['2deg']['D65'], sigfig=0),
            overwrite=True
        )

        cct1, duv1 = Color('orange').cct()
        assert math.isclose(cct1, 2423.929632254136, rel_tol=1e-11, abs_tol=1e-11)
        assert math.isclose(duv1, 0.00811119780855307, rel_tol=1e-11, abs_tol=1e-11)
        cct2, duv2 = Custom('orange').cct()
        assert math.isclose(cct2, 2423.930481644873, rel_tol=1e-11, abs_tol=1e-11)
        assert math.isclose(duv2, 0.008112876273860207, rel_tol=1e-11, abs_tol=1e-11)

    def test_robertson_discontinuity(self):
        """Test logic around the discontinuity in Robertson approach."""

        cct = 1625
        duv = 0.02
        cct2, duv2 = Color.blackbody('xyz-d65', cct, duv, scale=False).cct()

        assert math.isclose(cct, cct2, rel_tol=0.00001, abs_tol=0.00001)
        assert math.isclose(duv, duv2, rel_tol=0.00001, abs_tol=0.00001)


class TestRobertsonDuvConvention(util.ColorAssertsPyTest):
    """
    Test the sign of Duv against the documented convention.

    The documentation states that positive Duv is above the Planckian curve (the larger
    `v` side) and negative is below it. Both isotherm normals are perpendicular to the
    locus, and the stored one is the copy facing increasing `u`. Because the locus has a
    maximum in `v`, that stored copy is only "above" on one side of that vertex (between
    mired 600 and 625, i.e. 1600K - 1666.7K for the default table), which is what made the
    reading sign invert below it. These tests pin the whole band, both band rims, the
    vertex itself, custom tables and cross-method agreement.
    """

    def test_band_matrix(self):
        """Every in-band temperature/sign combination must round trip with an unchanged sign."""

        for cct in list(range(1000, 1667, 25)) + [1666, 1666.6, 1666.66, 1666.666, 1667]:
            for duv in (-0.05, -0.03, -0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.03, 0.05):
                cct2, duv2 = Color.blackbody(
                    'xyz-d65', cct, duv, scale=False, method='robertson-1968'
                ).cct(method='robertson-1968')
                assert math.isclose(cct2, cct, rel_tol=1e-6, abs_tol=1e-6)
                assert math.isclose(duv2, duv, rel_tol=1e-6, abs_tol=1e-9)
                if duv:
                    assert (duv2 > 0) == (duv > 0), f'sign inverted at {cct}K, Duv {duv}'

    def test_positive_duv_is_above_curve(self):
        """Positive Duv must place the color on the larger `v` side of the locus."""

        for cct in (1000, 1200, 1500, 1599, 1600, 1601, 1625, 1666, 1667, 2000, 5000, 20000):
            for method in ('robertson-1968', 'ohno-2013'):
                locus = Color.blackbody('xyz-d65', cct, 0.0, scale=False, method=method).uv('1960')[1]
                plus = Color.blackbody('xyz-d65', cct, 0.02, scale=False, method=method).uv('1960')[1]
                minus = Color.blackbody('xyz-d65', cct, -0.02, scale=False, method=method).uv('1960')[1]
                assert minus < locus < plus, f'{method} at {cct}K: {minus} < {locus} < {plus} violated'

    def test_lower_rim(self):
        """At the bottom of the table (1000K) the sign must be correct on both sides."""

        for cct in (999.9, 1000.0, 1000.1):
            for duv in (-0.05, -0.03, -0.02, -0.01, -0.001, 0.001, 0.01, 0.02, 0.03, 0.05):
                duv2 = Color.blackbody(
                    'xyz-d65', cct, duv, scale=False, method='robertson-1968'
                ).cct(method='robertson-1968')[1]
                assert math.isclose(duv2, duv, rel_tol=1e-6, abs_tol=1e-9)
                assert (duv2 > 0) == (duv > 0)

    def test_below_table_rim(self):
        """
        Temperatures below the table bottom clamp to the last bracket but keep the sign.

        The temperature is re-derived from the clamped bracket, so it is only accurate to
        the bracket spacing; Duv is what this test pins.
        """

        for cct in (500, 700, 900, 950, 999):
            for duv in (-0.03, -0.01, 0.01, 0.03):
                cct2, duv2 = Color.blackbody(
                    'xyz-d65', cct, duv, scale=False, method='robertson-1968'
                ).cct(method='robertson-1968')
                assert math.isclose(cct2, cct, rel_tol=1e-4, abs_tol=1e-2)
                assert math.isclose(duv2, duv, rel_tol=1e-4, abs_tol=1e-6)
                assert (duv2 > 0) == (duv > 0)

    def test_vertex_rim(self):
        """
        At the locus vertex (exactly 1600K in the default table) the sign must hold at all magnitudes.

        These are the rims of the vertex: the bracket used for generation (mired [625, 650])
        and the one chosen for reading (mired [600, 625] or [625, 650], depending on the
        direction and magnitude of the offset) disagree, which is where the old per-pair sign
        guess inverted reading against generation.
        """

        for duv in [s * i * 0.001 for i in range(1, 51) for s in (1, -1)] + [-0.05, 0.05]:
            cct2, duv2 = Color.blackbody(
                'xyz-d65', 1600.0, duv, scale=False, method='robertson-1968'
            ).cct(method='robertson-1968')
            assert math.isclose(cct2, 1600.0, rel_tol=1e-6, abs_tol=1e-6)
            assert math.isclose(duv2, duv, rel_tol=1e-6, abs_tol=1e-9)
            assert (duv2 > 0) == (duv > 0)

    def test_vertex_continuity(self):
        """Reading across the vertex must not jump in Duv."""

        prev = None
        u = 0.335
        while u <= 0.355:
            duv = Color.chromaticity('xyz-d65', [u, 0.3705], 'uv-1960', scale=False).cct(
                method='robertson-1968'
            )[1]
            if prev is not None:
                assert abs(duv - prev) < 1e-4, f'jump at u={u}: {prev} -> {duv}'
            prev = duv
            u += 0.0001

    def test_seam_between_slopes(self):
        """The mired 600/625 seam (the last slope sign change) must not disturb either side."""

        for cct in (1599.0, 1599.9, 1600.0, 1600.1, 1601.0, 1666.0, 1666.6, 1666.66, 1666.666, 1667.0, 1700.0):
            for duv in (-0.03, -0.01, -0.001, 0.001, 0.01, 0.03):
                duv2 = Color.blackbody(
                    'xyz-d65', cct, duv, scale=False, method='robertson-1968'
                ).cct(method='robertson-1968')[1]
                assert math.isclose(duv2, duv, rel_tol=1e-5, abs_tol=1e-8)
                assert (duv2 > 0) == (duv > 0)

    def test_cross_method_agreement(self):
        """Both registered methods must agree in sign (and closely in magnitude) on the same color."""

        for cct in (1000, 1200, 1500, 1599, 1600, 1601, 1625, 1667, 2000, 2500, 3000, 5000, 6500):
            for duv in (-0.03, -0.02, -0.01, 0.01, 0.02, 0.03):
                color = Color.blackbody('xyz-d65', cct, duv, scale=False, method='ohno-2013')
                cct_r, duv_r = color.cct(method='robertson-1968')
                cct_o, duv_o = color.cct(method='ohno-2013')
                assert math.isclose(cct_r, cct_o, rel_tol=1e-3, abs_tol=1e-3)
                assert math.isclose(duv_r, duv_o, abs_tol=1e-3)
                assert (duv_r > 0) == (duv_o > 0) == (duv > 0)

    def test_high_band_fidelity(self):
        """Temperatures at and above the original Robertson range must be unchanged by the fix."""

        anchors = (
            (1667, 0.02, 1667.000000828936, 0.020000000000731332),
            (5000, 0.02, 5000.000000000001, 0.020000000000000042),
            (20000, -0.03, 19999.99999999997, -0.029999999999999985),
            (100000, 0.0, 100000.00000000029, 2.689627682123609e-17)
        )
        for cct, duv, cct2, duv2 in anchors:
            got = Color.blackbody('xyz-d65', cct, duv, scale=False, method='robertson-1968').cct(
                method='robertson-1968'
            )
            assert math.isclose(got[0], cct2, rel_tol=1e-11, abs_tol=1e-11)
            assert math.isclose(got[1], duv2, rel_tol=1e-11, abs_tol=1e-11)

    def test_from_cct_fidelity(self):
        """`from_cct` must keep generating the same chromaticities; only the reading sign changes."""

        anchors = (
            (1000, 0.02, 0.4497373744950235, 0.3745452648000959),
            (1600, 0.01, 0.34513269185992956, 0.3705298611774313),
            (1600, -0.01, 0.34502730814007043, 0.3505301388225687),
            (2500, 0.02, 0.26771214966157947, 0.3735645713816302),
            (5000, -0.03, 0.2324410493762589, 0.30171636752509323),
            (100000, 0.0, 0.18066, 0.26589)
        )
        for cct, duv, u, v in anchors:
            uv = Color.blackbody('xyz-d65', cct, duv, scale=False, method='robertson-1968').uv('1960')
            assert math.isclose(uv[0], u, rel_tol=1e-12, abs_tol=1e-12)
            assert math.isclose(uv[1], v, rel_tol=1e-12, abs_tol=1e-12)

    def test_custom_table_vertex(self):
        """
        The sign band must follow each table's own slope sign change, not a hard coded 1600K.

        A table at a 10 mired resolution puts the vertex in the [610, 620] mired interval
        (~1613K - 1639K), so a hard coded temperature would still invert part of that band.
        """

        from coloraide.temperature.robertson_1968 import Robertson1968
        from coloraide import cat

        plugin = Robertson1968(cmfs.CIE_1931_2DEG, cat.WHITES['2deg']['D65'], tuple(range(0, 1001, 10)))
        vertex = [
            current.mired
            for index, current in enumerate(plugin.table)
            if index and current.slope * plugin.table[index - 1].slope < 0
        ]
        assert vertex == [620.0]

        for cct in range(1000, 2001, 20):
            for duv in (-0.03, -0.01, 0.01, 0.03):
                uv, name = plugin.from_cct(cct, duv)
                color = Color.chromaticity('xyz-d65', uv, name, scale=False)
                duv2 = plugin.to_cct(color)[1]
                assert (duv2 > 0) == (duv > 0), f'inverted at {cct}K, Duv {duv}: {duv2}'

    def test_custom_white_point_matches(self):
        """A D50 table must obey the same convention and round trip as the default D65 table."""

        from coloraide.temperature.robertson_1968 import Robertson1968
        from coloraide import cat

        plugin = Robertson1968(cmfs.CIE_1931_2DEG, cat.WHITES['2deg']['D50'])
        for cct in (1000, 1200, 1400, 1600, 1620, 1667, 2000, 5000):
            for duv in (-0.03, -0.01, 0.0, 0.01, 0.03):
                uv, name = plugin.from_cct(cct, duv)
                color = Color.chromaticity('xyz-d65', uv, name, scale=False)
                cct2, duv2 = plugin.to_cct(color)
                assert math.isclose(cct2, cct, rel_tol=1e-6, abs_tol=1e-6)
                assert math.isclose(duv2, duv, rel_tol=1e-6, abs_tol=1e-8)
                if duv:
                    assert (duv2 > 0) == (duv > 0)


