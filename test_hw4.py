import os
import ast
import inspect
import unittest
import numpy as np


from test_vars import X, y, test2_predictions, test3_coefs, test3_predictions

# import homework
import hw4


X = np.array(X)
y = np.array(y)


def uses_loop(function):
    for node in ast.walk(ast.parse(inspect.getsource(function))):
        if isinstance(node, ast.Name) and node.id == "map":
            return True
        elif isinstance(node, (ast.For, ast.While, ast.ListComp)):
            return True


class Test1CostFunction(unittest.TestCase):
    def test010_cost_grad(self):
        _, cols = X.shape
        theta0 = np.ones(cols)

        grad = hw4.gradient(X, y, theta0)

        def cost_(theta):
            return hw4.cost(X, y, theta)

        eps = 10**-4
        theta0_ = theta0
        grad_num = np.zeros(grad.shape)
        for i in range(grad.size):
            theta0_[i] += eps
            h = cost_(theta0_)
            theta0_[i] -= 2 * eps
            l = cost_(theta0_)
            theta0_[i] += eps
            grad_num[i] = (h - l) / (2 * eps)

        np.testing.assert_almost_equal(grad, grad_num, decimal=3)

    def test020_cost_function_vectorized(self):
        self.assertFalse(
            uses_loop(hw4.cost), "Implementation of cost function is not vectorized."
        )

    def test030_gradient_vectorized(self):
        self.assertFalse(
            uses_loop(hw4.gradient), "Implementation of gradient is not vectorized."
        )


class Test2LinearRegression(unittest.TestCase):
    def test010_vectorized(self):
        self.assertFalse(
            uses_loop(hw4.LinearRegression),
            "Methods in LR class should not have loops.",
        )

    def test020_test_univariate(self):
        train_x = X[:140, :1]
        train_y = y[:140]

        new_x = X[140:, :1]

        lr = hw4.LinearRegression()
        lr.fit(train_x, train_y)

        self.assertIsNotNone(lr.coefs)
        self.assertIsNotNone(lr.intercept)

        np.testing.assert_almost_equal(lr.coefs, [69.98018352], decimal=3)
        self.assertAlmostEqual(lr.intercept, 16.10088543792357, places=3)

        # test predictions
        np.testing.assert_almost_equal(lr.predict(new_x), test2_predictions, decimal=3)

    def test030_test_multivariate(self):
        train_x = X[:140]
        train_y = y[:140]

        new_x = X[140:]

        lr = hw4.LinearRegression()
        lr.fit(train_x, train_y)

        self.assertIsNotNone(lr.coefs)
        self.assertIsNotNone(lr.intercept)

        np.testing.assert_almost_equal(lr.coefs, test3_coefs, decimal=3)
        self.assertAlmostEqual(lr.intercept, -1.6506122502201706, places=3)

        np.testing.assert_almost_equal(lr.predict(new_x), test3_predictions, decimal=3)


class Test3LinearRegressionExample(unittest.TestCase):
    def test_file_reader(self):
        X, y = hw4.file_reader("data.tab")
        self.assertEqual(X.shape, (100, 10))
        self.assertEqual(y.shape, (100,))

    def test_images(self):
        image1 = "image1.png"
        image2 = "image2.png"

        self.assertIn(image1, os.listdir("."))
        self.assertIn(image2, os.listdir("."))


if __name__ == "__main__":
    unittest.main(verbosity=2)
