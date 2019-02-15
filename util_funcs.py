import numpy as np

def sigmoid(z):
    """
    the sigmoid function. :math:`\\frac{1}{1 + e^{-z}}`

    :param z: int or float or array-like; input of the function
    :return: float or ndarray
    """
    return 1 / (1 + np.exp(-z))



if __name__ == '__main__':
    tau = sigmoid(np.arange(10))