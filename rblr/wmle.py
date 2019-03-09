import numpy as np
from rblr.huber import Huber
from sklearn.covariance import MinCovDet
from rblr.util_funcs import sigmoid
from scipy.optimize import fmin_l_bfgs_b
from rblr.simulation_setup import simulation_setup
from sklearn.preprocessing import StandardScaler
from rblr.mestimator import MEstimator
import time


class WMLE:

    def __init__(self, clipping=1.345, fit_intercept=True, warm_start=False):
        self.huber = Huber(clipping=clipping)
        self.beta = None
        self.warm_start_flag = warm_start
        if warm_start:
            self.beta_last_fit = []
        self.fit_intercept = fit_intercept
        self.intercept_ = [0]
        self.coef_ = None
        self.lbfgs_warnflag_record = 1

    def set_beta(self, beta):
        self.beta = beta

    def leverage(self, X):
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

    def weights_factor(self, X):
        leverage = self.leverage(X)
        m_estimator = MEstimator()
        loc = m_estimator.loc_estimator(leverage)
        return self.huber.m_weights(leverage - loc)

    def probability(self, beta, X):
        return sigmoid(np.dot(X, beta))

    def cost_function(self, beta, X, y):
        # add an small offset to avoid Runtimewarning in log function
        epsilon = 1e-5
        n = X.shape[0]
        # X is concatenated with one column of 1s
        weights = self.weights_factor(X[:, 1:])

        total_cost = - (1/n) * np.sum(weights[:, np.newaxis] * (
            y * np.log(self.probability(beta, X) + epsilon) +
            (1 - y) * np.log(1 - self.probability(beta, X) + epsilon)))
        return total_cost

    def gradient(self, beta, X, y):
        n = X.shape[0]
        # X is concatenated with one column of 1s
        weights = self.weights_factor(X[:, 1:])
        X = X * weights[:, np.newaxis]
        return (1/n) * np.dot(np.transpose(X), sigmoid(np.dot(X, beta)) - y)

    def fit(self, X, y):
        m_estimator = MEstimator()
        loc = m_estimator.loc_estimator(arr=X, axis=0)
        scale = m_estimator.scale_estimator(arr=X, axis=0)
        X = (X - loc)/scale

        # add one column of 1s
        X = np.concatenate((np.ones((X.shape[0], 1), dtype=int), X), axis=1)

        if not self.warm_start_flag:
            x0 = np.ones(X.shape[1])
        else:
            if (len(self.beta_last_fit) == 0) or (self.lbfgs_warnflag_record != 0):
                x0 = np.ones(X.shape[1])
            else:
                x0 = self.beta_last_fit.pop()

        optimal = fmin_l_bfgs_b(func=self.cost_function,
                            x0=x0,
                            fprime=self.gradient,
                            args=(X, y))
        self.lbfgs_warnflag_record = optimal[-1]['warnflag']

        if self.warm_start_flag:
            self.beta_last_fit.append(optimal[0])
        self.beta = optimal[0]
        self.intercept_[0] = self.beta[0]
        self.coef_ = self.beta[1:]
        return self



    def predict(self, X, prob_threshold):
        if self.beta is None:
            raise ValueError("MLE Model is not fitted yet")
        ss = StandardScaler()
        X = ss.fit_transform(X)
        X = np.concatenate((np.ones((X.shape[0], 1), dtype=int), X), axis=1)
        predict_prob = self.probability(self.beta, X)
        return np.array(predict_prob >= prob_threshold, dtype=int)

    def score(self, X_test, y_test, prob_threshold=0.5):
        y_test = np.array(y_test, dtype=int)
        y_predicted = self.predict(X_test, prob_threshold)
        accuracy = np.mean(y_predicted == y_test)
        return accuracy

if __name__ == '__main__':
    # wmle = WMLE(fit_intercept=True)
    # t1 = time.time()
    # wmle.fit(X_train, y_train)
    #
    # lr = LogisticRegression(fit_intercept=True, solver='lbfgs')
    # lr.fit(X_train, y_train)
    #
    # print("WMLE accuracy: ", wmle.score(X_test, y_test))
    # print("Classical LR accuracy: ", lr.score(X_test, y_test))
    # print('wmle consumed time: %.5f s' % (time.time() - t1))


    n = 5
    accuracy_arr = np.zeros(n)
    t1 = time.time()
    for i in range(n):
        np.random.seed()
        X_train, y_train, X_test, y_test = simulation_setup(
            n_i=1000, n_o=0, n_t=1000, p=10, sigma_e=0.25)
        wmle = WMLE(fit_intercept=True, warm_start=False)
        wmle.fit(X_train, y_train)
        accuracy_arr[i] = wmle.score(X_test, y_test)
    print("elapsed time: %.2f s" % (time.time() - t1))
    accuracy_arr.sort()
