import math
import unittest

import main


class InitialValueProblemTests(unittest.TestCase):
    def test_exact_solution_satisfies_initial_conditions(self):
        self.assertAlmostEqual(main.cauchy_exact(1.0), 1.0)
        h = 1e-5
        numerical_derivative = (main.cauchy_exact(1 + h) - main.cauchy_exact(1 - h)) / (2 * h)
        self.assertAlmostEqual(numerical_derivative, 1.0, places=8)

    def test_runge_kutta_refinement_reduces_error(self):
        _, coarse = main.runge_kutta_method(main.cauchy_system, 1, 2, (1, 1), 0.1)
        _, fine = main.runge_kutta_method(main.cauchy_system, 1, 2, (1, 1), 0.05)
        exact = main.cauchy_exact(2)
        self.assertLess(abs(fine[-1][0] - exact), abs(coarse[-1][0] - exact))

    def test_adams_starts_with_runge_kutta(self):
        _, adams = main.adams_method(main.cauchy_system, 1, 2, (1, 1), 0.1)
        _, rk4 = main.runge_kutta_method(main.cauchy_system, 1, 2, (1, 1), 0.1)
        for index in range(4):
            self.assertAlmostEqual(adams[index][0], rk4[index][0])
            self.assertAlmostEqual(adams[index][1], rk4[index][1])


class BoundaryValueProblemTests(unittest.TestCase):
    def test_exact_solution_satisfies_boundary_conditions(self):
        h = 1e-5

        def derivative(x):
            return (main.boundary_exact(x + h) - main.boundary_exact(x - h)) / (2 * h)

        self.assertAlmostEqual(derivative(1), 1.5, places=8)
        self.assertAlmostEqual(
            main.boundary_exact(2) + 4 * derivative(2),
            13 + 4 * math.log(2),
            places=8,
        )

    def test_shooting_meets_right_boundary_condition(self):
        _, _, _, residual = main.shooting_method(0.1)
        self.assertLess(abs(residual), 1e-11)

    def test_finite_difference_refinement_reduces_error(self):
        xs_coarse, coarse = main.finite_difference_method(0.1)
        xs_fine, fine = main.finite_difference_method(0.05)
        error_coarse = max(abs(y - main.boundary_exact(x)) for x, y in zip(xs_coarse, coarse))
        error_fine = max(abs(y - main.boundary_exact(x)) for x, y in zip(xs_fine, fine))
        self.assertLess(error_fine, error_coarse)

    def test_runge_romberg_rejects_incompatible_grids(self):
        with self.assertRaises(ValueError):
            main.runge_romberg([1, 2], [1, 1.5], 2)


if __name__ == "__main__":
    unittest.main()
