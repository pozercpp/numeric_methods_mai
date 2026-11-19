import unittest

import main


class NumericalMethodsTests(unittest.TestCase):
    def test_lagrange_and_newton_are_equal_at_nodes(self):
        xs = [0.1, 0.5, 0.9, 1.3]
        ys = [1 / x**2 + x**2 for x in xs]
        lagrange = main.lagrange_polynomial(xs, ys)
        newton = main.newton_to_standard(xs, main.divided_differences(xs, ys))
        for x, y in zip(xs, ys):
            self.assertAlmostEqual(main.poly_eval(lagrange, x), y)
            self.assertAlmostEqual(main.poly_eval(newton, x), y)

    def test_natural_spline_interpolates_nodes(self):
        xs = [0.1, 0.5, 0.9, 1.3, 1.7]
        ys = [100.01, 4.25, 2.0446, 2.2817, 3.236]
        spline = main.natural_cubic_spline(xs, ys)
        for x, y in zip(xs, ys):
            self.assertAlmostEqual(main.spline_eval(xs, spline, x), y)

    def test_known_results(self):
        xs = [0.1, 0.5, 0.9, 1.3, 1.7, 2.1]
        ys = [100.01, 4.25, 2.0446, 2.2817, 3.236, 4.6368]
        matrix, rhs = main.normal_equations(xs, ys, 2)
        coefficients = main.solve_linear_system(matrix, rhs)
        expected = [98.44974527, -156.64771607, 55.62977679]
        for actual, wanted in zip(coefficients, expected):
            self.assertAlmostEqual(actual, wanted, places=7)

    def test_simpson_refines_integral(self):
        function = lambda x: x**2 * (36 - x**2) ** 0.5
        coarse = main.composite_simpson(function, 1, 5, 1)
        fine = main.composite_simpson(function, 1, 5, 0.5)
        refined, _ = main.runge_romberg(coarse, fine, 4)
        self.assertAlmostEqual(refined, 186.6202575152, places=8)


if __name__ == "__main__":
    unittest.main()
