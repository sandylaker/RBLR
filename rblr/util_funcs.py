import numpy as np
from scipy.special import expit
from sklearn.covariance import MinCovDet

def sigmoid(z):
    """
    the sigmoid function. :math:`\\frac{1}{1 + e^{-z}}`

    :param z: int or float or array-like; input of the function
    :return: float or ndarray
    """
    return expit(z)

def deviance_residual(y, log_prob):
    """
    calculate the deviance residuals of fitted model

    :param y: array-like,
                label
    :param log_prob: array, shape = [n_samples, 2]
                estimated log probability of each class, the log probability all all
                classes are ordered by the labels of classes, i.e. [0,1]
    :return: array, shape = [n_samples,]
                deviance residuals
    """
    if not (np.unique(y) == np.arange(2)).all():
        raise ValueError("y must be encoded as 1 or 0")
    return (2 * y - 1) * np.sqrt(- 2 * (y * log_prob[:, 0] + (1 - y) * log_prob[:, 1]))

def leverage(X):
    mcd = MinCovDet()
    mcd.fit(X)
    loc, cov = mcd.location_, mcd.covariance_
    inversed_cov = np.linalg.inv(cov)
    result = np.zeros(X.shape[0])
    for i, element in enumerate(X):
        h = np.sqrt(
            np.transpose(element - loc) @ inversed_cov @ (element - loc))
        result[i] = h
    return result


if __name__ == '__main__':
    tau = sigmoid(np.arange(10))
    X = np.random.randint(0, 100, (10, 10))
    l = leverage(X)