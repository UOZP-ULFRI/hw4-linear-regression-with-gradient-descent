import numpy as np
import matplotlib.pyplot as plt


from typing import Optional, Tuple


def cost(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """A cost function in matrix/vector form.

    Parameters
    ----------
    X: numpy array of shape (n_samples, n_features)
        Training data
    y: numpy array of shape (n_samples,)
        Target values
    theta: numpy array of shape (n_features,)
        Parameters

    Returns
    -------
    float
    """
    raise NotImplementedError


def gradient(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Gradient of cost function in matrix/vector form.

    Parameters
    ----------
    X: numpy array of shape (n_samples, n_features)
        Training data
    y: numpy array of shape (n_samples,)
        Target values
    theta: numpy array of shape (n_features,)
        Parameters

    Returns
    -------
    numpy array of shape (n_features,)
    """
    raise NotImplementedError


def gradient_descent(
    X: np.ndarray, y: np.ndarray, lr=0.01, tol=1e-7, max_iter=10_000
) -> np.ndarray:
    """Implementation of gradient descent.

    Parameters
    ----------
    X: numpy array of shape (n_samples, n_features)
        Training data
    y: numpy array of shape (n_samples,)
        Target values
    lr: float
        The learning rate.
    tol: float
        The stopping criterion (tolerance).
    max_iter: int
        The maximum number of passes (aka epochs).

    Returns
    -------
    numpy array of shape (n_features,)
    """
    raise NotImplementedError


class LinearRegression:
    def __int__(self) -> None:
        self.coefs: Optional[np.ndarray] = None
        self.intercept: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        The fit method of LinearRegression accepts X and y
        as input and save the coefficients of the linear model.

        Parameters
        ----------
        X: numpy array of shape (n_samples, n_features)
            Training data
        y: numpy array of shape (n_samples,)
            Target values

        Returns
        -------
        None
        """
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the linear model.

        Parameters
        ----------
        X: numpy array of shape (n_samples, n_features)
            New samples

        Returns
        -------
        numpy array of shape (n_samples,)
            Returns predicted values.
        """
        raise NotImplementedError


def file_reader(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    file_path: str
        path to data.tab

    Returns
    -------
    X: numpy array of shape (n_samples, n_features)
        Training data
    y: numpy array of shape (n_samples,)
        Target values
    """
    raise NotImplementedError


if __name__ == "__main__":
    raise NotImplementedError
