from rblr.util_funcs import leverage
from sklearn.linear_model import LogisticRegression
from rblr import Huber, MEstimator
import numpy as np
import matplotlib.pyplot as plt

class FastWMLE(LogisticRegression):

    def __init__(self, clipping=1.345, solver='lbfgs', **kwargs):
        self.huber = Huber(clipping=clipping)
        super().__init__(solver=solver, **kwargs)

    def weights_factor(self, X):
        leverage_arr = leverage(X)
        m_estimator = MEstimator()
        loc = m_estimator.loc_estimator(leverage_arr)
        return self.huber.m_weights(leverage_arr - loc)

    def fit(self, X, y, **kwargs):
        weights = self.weights_factor(X)
        super().fit(X, y, sample_weight=weights)
        return self


if __name__ == '__main__':
    n = 100
    score_arr = np.zeros(n)

    for i in range(n):
        np.random.seed()
        X_train, y_train, X_test, y_test = simulation_setup(n_i=1000, n_o=0, n_t=1000, p=10,
                                                            sigma_e=0.25)
        fast_wmle = FastWMLE(max_iter=500)
        fast_wmle.fit(X_train, y_train)
        # ss = StandardScaler()
        # X_test= ss.fit_transform(X_test)
        score_arr[i] = fast_wmle.score(X_test, y_test)

    plt.hist(score_arr, bins=10)
    plt.show()