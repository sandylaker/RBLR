import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from util_funcs import sigmoid

class IFB:

    def __init__(self, c=None, gamma=10, fit_intercept=True):
        """
        the robust Logistic Regression model based on influence function bootstrap(IFB).

        :param c: double, optional,
            turning constant of the psi function in calculating weights in IFB.

        :param gamma: double, default: 10,
            a parameter in the psi function in calculating weights in IFB.

        :param fit_intercept: bool, default: True.
            specifies if a constant(intercept) should be added to the decision function.
        """
        self.c = c
        self.gamma = gamma
        self.beta = None
        self.fit_intercept = fit_intercept
        self.intercept_ = None
        self.coef_ = None

    def resif(self, X):
        """
        computes the robust empirical influence function(RESIF). Choose to use :math:`\Omega_2 =
        ( \\theta, \\hat{\Sigma} )` as estimator of central model.

        :param X: ndarray, shape(n_samples, n_features)
                Training data

        :return: ndarray, shape(n_samples,)
                RESIF of each sample
        """
        mcd = MinCovDet()
        mcd.fit(X=X)
        loc, cov = mcd.location_, mcd.covariance_
        inversed_cov = np.linalg.inv(cov)
        result = np.zeros(len(X))
        for i, element in enumerate(X):
            h = np.sqrt(
                np.transpose(element - loc) @ inversed_cov @ (element - loc))
            result[i] = h

        return result

    def psi(self, x, c, gamma):
        """
        computes the psi function for calculating weights in IFB

        :param x: ndarray,
                input array

        :param c: double,
                turning constant

        :param gamma: double,
                a parameter

        :return: ndarray
        """
        if gamma > 0 and gamma != float('Inf'):
            return (1 + (x - c) ** 2 / (gamma * c ** 2)) ** (- 0.5 * (gamma + 1))
        elif gamma < 0:
            raise ValueError("gamma must be positive")
        elif gamma == float('Inf'):
            return np.exp(- (x - c) ** 2 / (2 * c ** 2))

    def weights(self, resif_array, **kwargs):
        """
        computes the weights in IFB

        :param resif_array: ndarray, shape(n_samples,),
                            array of RESIF

        :param kwargs: keyword arguments of psi function

        :return: ndarray, shape(n_samples,)
                array of weights of samples
        """
        c = kwargs['c']
        indicator_1 = np.logical_and(resif_array >= 0, resif_array <= c).astype(int)
        indicator_2 = np.greater_equal(resif_array, c)
        return indicator_1 + self.psi(resif_array, **kwargs) * indicator_2

    def sample_probability(self, X, quantile_factor=0.9):
        """
        computes the sampling probability of each sample.

        :param X: ndarray, shape(n_samples, n_features),
                Training data.

        :param quantile_factor: double, default: 0.9,
                                if the turning constant of psi function is not given, the turning
                                constant will be determined by the quantile factor, default will be
                                set to 0.9 quantile of the RESIF

        :return: ndarray, shape(n_samples,),
                array of sampling probability
        """
        resif_array = self.resif(X)
        if not self.c:
            # NOTICE the quantile factor is adjustable
            self.c = np.quantile(resif_array, quantile_factor)
        weights_array = self.weights(resif_array, c=self.c, gamma=self.gamma)
        return weights_array / np.sum(weights_array)

    def fit(self, X, y, n_bootstrap=50, n_samples_each_bootstrap=None, solver='lbfgs',
            standard_scaled=True,
            quantile_factor=0.9,
            **kwargs):
        """
        fit the robust logistic regression model based on IFB

        :param X: ndarray, shape(n_samples, n_features)
                Training data

        :param y: ndarray, shape(n_samples,)
                labels of training data

        :param n_bootstrap: int, default: 50
                number of bootstrap samples

        :param n_samples_each_bootstrap: int, optional
                number of samples in every bootstrap sample, by default, the same number of
                samples as the training data will be sampled in every bootstrap samples.

        :param solver: str, default: 'lbfgs',
                solver of the logistic regression model. See more solvers in the documentation in
                scikit-learn.

        :param standard_scaled: bool, default: True
                if True, the data will be standard scaled.

        :param quantile_factor: double, default: 0.9,
                the quantile_factor for determining the turning constant in computing the weights
                for sampling. By default, the turning constant is 0.9 quantile of the sorted RESIF
                array.

        :param kwargs: keyword arguments of logistic regression model

        :return: the fitted model. the coefficients of the model will be the average of
                coefficients of each fitted model trained by bootstrap sample.
        """

        p = self.sample_probability(X, quantile_factor=quantile_factor)

        if standard_scaled:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        y = np.asarray(y)
        if y.ndim == 1:
            y = y[:, np.newaxis]

        # concatenate X and y horizontally together for convenience
        X_concat = np.concatenate((X, y), axis=1)
        X_index = np.arange(X_concat.shape[0])
        if n_samples_each_bootstrap is None:
            # by default, sample the same number of samples as training data in every bootstrap
            # sample
            n_samples_each_bootstrap = X_concat.shape[0]

        # sample with this index will generate a matrix
        # of shape(N_bootstrap, N_sample_pro_bootstrap, p_feature + 1)
        # we will draw the same number of observations as X in every bootstrap sample
        sample_index = np.random.choice(X_index, size=(n_bootstrap, n_samples_each_bootstrap), p=p)
        bootstrap_samples = X_concat[sample_index]
        # print('bootstrap samples shape: ', bootstrap_samples.shape, '\n')

        self.beta = []
        for X_concat_b in bootstrap_samples:
            # split X and y
            X_b = X_concat_b[:, :-1]
            y_b = X_concat_b[:, -1]
            clf = LogisticRegression(fit_intercept=self.fit_intercept, solver=solver, **kwargs)
            clf.fit(X_b, y_b)
            if self.fit_intercept:
                self.beta.append(np.concatenate((clf.intercept_, clf.coef_[0])))
            else:
                self.beta.append(clf.coef_[0])

        self.beta = np.mean(self.beta, axis=0)
        # normalize the average of coefficients
        self.beta = self.beta/np.linalg.norm(self.beta)
        if self.fit_intercept:
            self.intercept_ = self.beta[1:]
        self.coef_ = self.beta[self.fit_intercept:]
        return self

    # TODO predict function and score function not implemented
    def predict(self, X, prob_threshold=0.5):
        """
        predict labels for test data

        :param X: ndarray, shape(n_samples, n_features)
                        data to be predicted

        :param prob_threshold: double, default: 0.5,
                            probability threshold for determining the predicted labels.

        :return: ndarray, shape(n_samples,)
                predicted labels, which are coded with {1,0}
        """
        ss = StandardScaler()
        X = ss.fit_transform(X)
        if self.beta is None:
            raise ValueError("Model is not fitted yet")
        if self.fit_intercept:
            X = np.concatenate((np.ones((X.shape[0], 1), dtype=int), X), axis=1)
        predict_prob = sigmoid(np.dot(X, self.beta))
        return np.array(predict_prob >= prob_threshold, dtype=int)

    def score(self, X_test, y_test, prob_threshold=0.5):
        """
        computes accuracy of the model

        :param X_test: ndarray, shape(n_samples, n_features)
                    Test data

        :param y_test: ndarray, shape(n_samples,)
                    Labels of test data

        :param prob_threshold: double, default: 0.5
                    probability threshold for determining the predicted labels

        :return: double
                accuracy of the logistic regression model.
        """
        y_test = np.array(y_test, dtype=int)
        y_predict = self.predict(X_test, prob_threshold=prob_threshold)
        accuracy = np.mean(y_predict == y_test)
        return accuracy



if __name__ == '__main__':
    X = np.random.randint(0, 10, (100, 10))
    X = np.concatenate((X, np.random.randint(10, 20, (10, 10))), axis=0)
    ifb = IFB()
    robust_mcd = ifb.resif(X)
    psi_array = ifb.psi(robust_mcd, 7, 10)

    weights_array = ifb.weights(robust_mcd, c=np.quantile(robust_mcd, 0.9), gamma=10)
    sample_prob = ifb.sample_probability(X)

    # plt.figure(1)
    # plt.hist(robust_mcd, bins=50)
    # plt.title('robust_mcd')
    #
    # plt.figure(2)
    # plt.hist(psi_array, bins=50)
    # plt.title('psi_array')
    #
    # plt.figure(3)
    # plt.hist(weights_array, bins=50)
    # plt.title('weights_array')
    #
    # plt.figure(4)
    # plt.hist(sample_prob, bins=50)
    # plt.title('sample_prob')
    # plt.show()

