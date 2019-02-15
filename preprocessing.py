import numpy as np
import pandas as pd
from mestimator import MEstimator
from simulation_setup import simulation_setup

class Preprocessor(MEstimator):

    def __init__(self, clipping=4.586):
        super().__init__(clipping=clipping)

    def preprocessing(self, X, y=None, return_outliers=False):
        if type(X) == pd.DataFrame:
            X = X.values
        if type(y) == pd.DataFrame or type(y) == pd.Series:
            y = y.values
        X= np.asarray(X)
        if not y is None:
            y = np.asarray(y)
        n_total = X.shape[0]
        p = X.shape[1]

        loc_estimated = self.loc_estimator(X)
        scale_estimated = self.scale_estimator(X)
        X_centered = X - loc_estimated
        X_norm = np.linalg.norm(X_centered, axis=1)

        # NOTICE: here n_inlier should be used, however, the number of inliers is
        # NOTICE: usually unkown, so we use 0.5*n_total to estimate the maximal number of inliers.
        T = 4 * np.sqrt(np.log(p) + np.log(0.5 * n_total)) * np.max(scale_estimated)

        inlier_flag = np.less_equal(X_norm, T)
        X_in = X[inlier_flag]

        X_out = X[~inlier_flag]

        if y is None:
            if return_outliers:
                return X_in, X_out
            else:
                return X_in
        else:
            y_in, y_out = y[inlier_flag], y[~inlier_flag]
            if return_outliers:
                return X_in, y_in, X_out, y_out
            else:
                return X_in, y_in



if __name__ == '__main__':
    data = simulation_setup(n_i=1000, n_o=500, p=10)[0]
    X, y = data[:, :-1], data[:, -1]
    preprocessor = Preprocessor()
    # X_in, y_in, X_out, y_out = preprocessor.preprocessing(X, y, return_outliers=True)
    #     # print('number of X_in: %d, y_in: %d' % (len(X_in), len(y_in)))
    #     # print('number of X_out: %d, y_out: %d' % (len(X_out), len(y_out)))
    X_in = preprocessor.preprocessing(X)
    print(X_in.shape)