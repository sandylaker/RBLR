import numpy as np
import pandas as pd
from util_funcs import sigmoid


def simulation_setup(n_i=1000,
                     n_o=100,
                     n_t=1000,
                     p=20,
                     mu_i=None,
                     sigma_i=1,
                     sigma_e=0.5,
                     sigma_o=10,
                     to_csv=False):
    """
    generate the train and test data for simulation. Train data of shape
    :math:`(n_i + n_o, p)` and test data of shape :math:`(n_t, p)` will be generated.
    By default, train data contains 1000 inliers and 100 outliers, where the inlier covariant matrix
    conform to multivariate normal distribution :math:`X_in \sim \\mathsf{N}(\\mathbf{0},\\mathbf{\sum_{in}})`,
    by default, the covariance matrix is a unit matrix :math:`\\mathbf{1}`. The labels of inliers are
    generated based on :math:`\\mathsf{P}(y_i=1) = H(\\mathbf{\\beta}^T \mathbf{x_i} + v_i)`,
    where :math:`H(z) = \\frac{1}{1+e^{-z}}` and :math:`v_i \sim N(0, \\sigma_e)` is the gaussian noise, the labels of
    test data are generated based on
    :math:`y_i = u(\\mathbf{\\beta}^T \\mathbf{x_i})`,where :math:`u(x) = 1 \\text{for} \\ x \\geq 0, 0 \\text{for}
    \\ x<0`, the labels of outliers are generated based on
    :math:`\\mathsf{P}(y_i=1) = u(-\\mathbf{\\beta}^T \mathbf{x_i})`. Then, the labels are concatenated onto
    the covariante matrix along axis 1(horizontally).


    :param n_i: int, number of inliers
    :param n_o: int, number of outliers
    :param n_t: int, number of test observation
    :param p: int, number of dimension
    :param mu_i: ndarray, location of the inliers
    :param sigma_i: float, standard deviation of the inliers
    :param sigma_e: float, standard deviation of the gaussian noise
    :param sigma_o: float, standard deviation of the outliers
    :param to_csv: boolean, if True, the generated train and test data will be written
                    into csv files
    :return: tuple, tuple containing ndarray of train data with shape(n_i+n_o,p+1), test data with shape(n_t,p+1),
            ndarray of beta. Here beta is the true value of the coefficients in the logistic model. The last columns
            of train and test data are labels of {1,0}
    """
    if mu_i is None:
        mu_i = np.zeros(p)
    if len(mu_i) != p:
        raise ValueError("length of mu_i must be equal to p")

    # sample beta, here size=1 will generate a 2-D array of shape(1,p),
    # the flat array will be used, so the first component is extracted
    beta = np.random.multivariate_normal(np.zeros(p), np.eye(p), size=1)
    # normalize beta
    beta = beta[0] / np.linalg.norm(beta)

    # sample X_i of good data
    X_i = np.random.multivariate_normal(mu_i, np.eye(p) * sigma_i, size=n_i)
    # gaussian noise of good samples
    noise_i = np.random.randn(n_i) * sigma_e

    # generate labels of good samples
    tau = sigmoid(np.dot(beta, X_i.T) + noise_i)
    marker = np.array(np.less_equal(np.random.uniform(0, 1, n_i), tau))
    y_i = np.ones(n_i)
    y_i[~marker] = 0
    y_i = y_i.astype(int)

    # test sample
    X_t = np.random.multivariate_normal(mu_i, np.eye(p) * sigma_i, size=n_t)
    # generate label of test samples
    y_t = np.greater_equal(
        np.dot(
            beta,
            X_t.T) +
        np.random.randn(n_t) *
        sigma_e, 0).astype(int)

    # sample X_o of outliers
    X_o = np.random.uniform(- sigma_o, sigma_o, (n_o, p))
    # gaussian noise of outliers
    noise_o = np.random.randn(n_o) * sigma_e
    # generate labels of outliers
    y_o = np.greater_equal(np.dot(- beta, X_o.T) + noise_o, 0).astype(int)

    # concatenate the data into a dataframe
    y_i = y_i[:, np.newaxis]
    df_i = pd.DataFrame(data=np.concatenate((X_i, y_i), axis=1))

    y_t = y_t[:, np.newaxis]
    df_t = pd.DataFrame(data=np.concatenate((X_t, y_t), axis=1))

    y_o = y_o[:, np.newaxis]
    df_o = pd.DataFrame(data=np.concatenate((X_o, y_o), axis=1))

    # concatenate the good data and oulier data into a dataframe for training
    df_train = pd.concat([df_i, df_o], ignore_index=True)

    if to_csv:
        # write into csv
        df_train.to_csv('data_train.csv', index=False)
        df_t.to_csv('data_test.csv', index=False)
        print('sucessfully written into csv')
    return df_train.values, df_t.values, beta


if __name__ == '__main__':
    data_train, data_test, beta = simulation_setup()
