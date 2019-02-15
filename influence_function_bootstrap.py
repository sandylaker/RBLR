import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from util_funcs import sigmoid

class IFB:

    def __init__(self, c=None, gamma=10, fit_intercept=False):
        self.c = c
        self.gamma = gamma
        self.beta = None
        self.fit_intercept = fit_intercept

    def resif(self, X):
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
        if gamma > 0 and gamma != float('Inf'):
            return (1 + (x - c) ** 2 / (gamma * c ** 2)) ** (- 0.5 * (gamma + 1))
        elif gamma < 0:
            raise ValueError("gamma must be positive")
        elif gamma == float('Inf'):
            return np.exp(- (x - c) ** 2 / (2 * c ** 2))

    def weights(self, resif_array, **kwargs):
        c = kwargs['c']
        indicator_1 = np.logical_and(resif_array >= 0, resif_array <= c).astype(int)
        indicator_2 = np.greater_equal(resif_array, c)
        return indicator_1 + self.psi(resif_array, **kwargs) * indicator_2

    def sample_probability(self, X, quantile_factor=0.9):
        resif_array = self.resif(X)
        if not self.c:
            # NOTICE the quantile factor is adjustable
            self.c = np.quantile(resif_array, quantile_factor)
        weights_array = self.weights(resif_array, c=self.c, gamma=self.gamma)
        return weights_array / np.sum(weights_array)

    def fit(self, X, y, n_bootstrap=50, solver='lbfgs', standard_scaled=True, quantile_factor=0.9, **kwargs):

        p = self.sample_probability(X, quantile_factor=quantile_factor)

        if standard_scaled:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        # concatenate X and y horizontally together for convenience
        X_concat = np.concatenate((X, y[:, np.newaxis]), axis=1)
        X_index = np.arange(X_concat.shape[0])

        # sample with this index will generate a matrix
        # of shape(N_bootstrap, N_sample_pro_bootstrap, p_feature + 1)
        # we will draw the same number of observations as X in every bootstrap sample
        sample_index = np.random.choice(X_index, size=(n_bootstrap, X_concat.shape[0]), p=p)
        bootstrap_samples = X_concat[sample_index]
        print('bootstrap samples shape: ', bootstrap_samples.shape)

        self.beta = []
        for X_concat_b in bootstrap_samples:
            # split X and y
            X_b = X_concat_b[:, :-1]
            y_b = X_concat_b[:, -1]
            clf = LogisticRegression(fit_intercept=self.fit_intercept, solver=solver, **kwargs)
            clf.fit(X_b, y_b)
            self.beta.append(clf.coef_[0])

        self.beta = np.mean(self.beta, axis=0)
        # normalize the average of coefficients
        self.beta = self.beta/np.linalg.norm(self.beta)
        return self

    # TODO predict function and score function not implemented
    def predict(self, X_predict, prob_threshold=0.5):
        ss = StandardScaler()
        X_predict = ss.fit_transform(X_predict)
        if self.beta is None:
            raise ValueError("Model is not fitted yet")
        predict_prob = sigmoid(np.dot(X_predict, self.beta))
        return np.array(predict_prob >= prob_threshold, dtype=int)

    def score(self, X_test, y_test, prob_threshold=0.5):
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

    plt.figure(1)
    plt.hist(robust_mcd, bins=50)
    plt.title('robust_mcd')

    plt.figure(2)
    plt.hist(psi_array, bins=50)
    plt.title('psi_array')

    plt.figure(3)
    plt.hist(weights_array, bins=50)
    plt.title('weights_array')

    plt.figure(4)
    plt.hist(sample_prob, bins=50)
    plt.title('sample_prob')
    plt.show()

