import unittest

try:
    from hyperparameter_optimization import bisection, bracket_root
except ModuleNotFoundError:
    bisection = bracket_root = None


def derivative(x: float) -> float:
    return x ** 2 - 2


@unittest.skipIf(bisection is None, "Required packages not installed")
class TestRootMethods(unittest.TestCase):
    def test_bisection_auto_bracket(self):
        a, b = 0.0, 1.0
        root, _ = bisection(derivative, a, b)
        self.assertAlmostEqual(root, 2 ** 0.5, places=4)

    def test_bracket_root(self):
        a, b = bracket_root(derivative, 0.0, 1.0)
        self.assertLess(derivative(a) * derivative(b), 0)


if __name__ == "__main__":
    unittest.main()
