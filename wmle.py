import numpy as np
from huber import Huber
from sklearn.covariance import MinCovDet
import matplotlib.pyplot as plt
from util_funcs import sigmoid
from scipy.optimize import fmin_tnc
from simulation_setup import simulation_setup
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import time
from numba import jit


class WMLE:

    def __init__(self, clipping=1.345):
        self.huber = Huber(clipping=clipping)
        self.beta = None

    def leverage(self, X):
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

    def weights_factor(self, X):
        leverage = self.leverage(X)
        return self.huber.m_weights(leverage)

    def probability(self, beta, X):
        return sigmoid(np.dot(X, beta))

    def cost_function(self, beta, X, y):
        # add an small offset to avoid Runtimewarning in log function
        epsilon = 1e-5
        n = X.shape[0]
        weights = self.weights_factor(X)
        total_cost = - (1/n) * np.sum(weights[:, np.newaxis] * (
            y * np.log(self.probability(beta, X) + epsilon) +
            (1 - y) * np.log(1 - self.probability(beta, X) + epsilon)))
        return total_cost

    def gradient(self, beta, X, y):
        n = X.shape[0]
        weights = self.weights_factor(X)
        X = X * weights[:, np.newaxis]
        return (1/n) * np.dot(np.transpose(X), sigmoid(np.dot(X, beta)) - y)

    def fit(self, X, y):
        # X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        self.beta = np.zeros(X.shape[1])
        ss = StandardScaler()
        X = ss.fit_transform(X)
        optimal = fmin_tnc(func=self.cost_function,
                            x0=self.beta,
                            fprime=self.gradient,
                            args=(X, y))
        self.beta = optimal[0] / np.linalg.norm(optimal[0])
        return self

    def predict(self, X_predict, prob_threshold):
        ss = StandardScaler()
        X_predict = ss.fit_transform(X_predict)
        if self.beta is None:
            raise ValueError("MLE Model is not fitted yet")
        # X_predict = np.concatenate((np.ones(X_predict.shape[0], 1), X_predict), axis=1)
        predict_prob = self.probability(self.beta, X_predict)
        return np.array(predict_prob >= prob_threshold, dtype=int)

    def score(self, X_test, y_test, prob_threshold=0.5):
        y_test = np.array(y_test, dtype=int)
        y_predicted = self.predict(X_test, prob_threshold)
        accuracy = np.mean(y_predicted == y_test)
        return accuracy


if __name__ == '__main__':
    data_train, data_test, beta_actual = simulation_setup(
        n_i=1000, n_o=20, n_t=1000, p=20, sigma_e= 0.5)
    X_train, y_train = data_train[:, :-1], data_train[:, -1].astype(int)
    X_test, y_test = data_test[:, :-1], data_test[:, -1].astype(int)
    wmle = WMLE()
    t1 = time.time()
    wmle.fit(X_train, y_train)
    print("wmle accuracy: ", wmle.score(X_test, y_test))
    print("wmle coefficients: \n", wmle.beta)
    print("norm of coefficients: ", np.sqrt(np.sum(wmle.beta ** 2)))
    print('wmle consumed time: %.5f s' % (time.time() - t1))

    lr = LogisticRegression(fit_intercept=False, solver='lbfgs')
    lr.fit(X_train, y_train)
    print("LR accuracy: ", lr.score(X_test, y_test))
    print("LR coeffcients: \n", lr.coef_[0])
