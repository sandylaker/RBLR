import numpy as np
import cvxpy as cp
from simulation_setup import simulation_setup
from util_funcs import sigmoid
from scipy.optimize import minimize, fmin_tnc
from sklearn.linear_model import LogisticRegression


class MLE:

    def __init__(self):
        self.beta = None

    def probability(self, beta, X):
        return sigmoid(np.dot(X, beta))

    def cost_function(self, beta, X, y):
        n = X.shape[0]
        total_cost = - (1/n) * np.sum(
            y * np.log(self.probability(beta, X)) +
            (1 - y) * np.log(1 - self.probability(beta, X)))
        return total_cost

    def gradient(self, beta, X, y):
        n = X.shape[0]
        return (1/n) * np.dot(np.transpose(X), sigmoid(np.dot(X, beta)) - y)

    def fit(self, X, y):
        # X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        self.beta = np.zeros(X.shape[1])
        optimal = fmin_tnc(func=self.cost_function,
                            x0=self.beta,
                            fprime=self.gradient,
                            args=(X, y))
        self.beta = optimal[0]
        return self

    def predict(self, X_test):
        if self.beta is None:
            raise ValueError("MLE Model is not fitted yet")
        # X_test = np.concatenate((np.ones(X_test.shape[0], 1), X_test), axis=1)
        return self.probability(self.beta, X_test)

    def accuracy(self, X_test, y_test, prob_threshold=0.5):
        y_test = np.array(y_test, dtype=int)
        y_predicted = (self.predict(X_test) >= prob_threshold).astype(int)
        accuracy = np.mean(y_predicted == y_test)
        return accuracy


if __name__ =='__main__':
    data_train, data_test, beta_actual = simulation_setup(n_i=1000, n_o=0, n_t=1000, p=20)
    X_train, y_train = data_train[:, :-1], data_train[:, -1].astype(int)
    X_test, y_test = data_test[:, :-1], data_test[:, -1].astype(int)
    mle = MLE()
    mle.fit(X_train, y_train)
    acc = mle.accuracy(X_test, y_test)
    print('accuracy:', acc)
    print('fitted coefficients: \n', mle.beta)
    print('actual coefficients: \n', beta_actual)
    lr = LogisticRegression(fit_intercept=False, solver='lbfgs')
    lr.fit(X_train, y_train)
    score = lr.score(X_test, y_test)
    print('LR accuracy: ', score)
    print('LR coefficients: \n', lr.coef_[0])
    print('RMSE: ', np.sqrt(np.mean((mle.beta - lr.coef_[0])**2)))







