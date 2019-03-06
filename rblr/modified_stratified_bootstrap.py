import numpy as np
from rblr import StratifiedBootstrap, LogisticRegression, \
    simulation_setup, deviance_residual, MEstimator, Huber


class ModifiedStraitifiedBootstrap(StratifiedBootstrap):

    def __init__(self, clipping=1.345, fit_intercept=True,solver='lbfgs', **kwargs):
        super().__init__(clipping=clipping, fit_intercept=fit_intercept)
        self.clf_lr = LogisticRegression(fit_intercept=fit_intercept,solver=solver, **kwargs)

    def fit(self, X, y, n_bootstrap=20, n_strata=5,
            metric='leverage',
            **kwargs):
        y = np.asarray(y).astype(int)
        # calculate the weights of each observation
        metric_arr = self._metric_arr(X, y, metric)
        weights_factor = self.weights_factor(X, y, metric_arr)

        if y.ndim == 1:
            y = y[:, np.newaxis]
        # concatenate weigts and X and y horizontally for convenience
        X_concat = np.concatenate((weights_factor[:, np.newaxis], X, y), axis=1)

        # sort the Xy matrix according to metric array and split into strata
        order = np.argsort(metric_arr)
        X_concat = X_concat[order]
        stratas = np.array_split(X_concat, n_strata)

        self.beta = []
        for i in range(n_bootstrap):
            for stratum in stratas:
                # in each stratum, resample the same number of samples as the stratum
                sample_index = np.random.choice(stratum.shape[0], stratum.shape[0], replace=True)
                X_concat_b = stratum[sample_index]
                # in X_concat_b: first column: weight; last column: y
                weight_average = np.mean(X_concat_b[:, 0])
                X_ = X_concat_b[:, 1:-1]
                y_ = X_concat_b[:, -1].astype(int)
                # check if there are two classes in the bootstrap sample
                if np.unique(y_).shape[-1] == 1:
                    raise ValueError("There is only one class of data in the bootstrap sample, so logistic regression "
                                     "model is unable to classify, please reduce number of strata")
                self.clf_lr.fit(X_, y_)
                beta = np.concatenate((self.clf_lr.intercept_, self.clf_lr.coef_[0]))
                self.beta.append(weight_average * beta)
        self.beta = np.mean(self.beta, axis=0)
        self.intercept_[0] = self.beta[0]
        self.coef_ = self.beta[1:]
        return self

    def _metric_arr(self, X, y, metric='residual'):
        if metric == 'residual':
            self.clf_lr.fit(X, y)
            log_prob = self.clf_lr.predict_log_proba(X)
            metric_arr = deviance_residual(y, log_prob)
        elif metric == 'leverage':
            metric_arr = self.wmle.leverage(X)
        else:
            raise ValueError("metric can be either residual or leverage")
        return metric_arr

    def weights_factor(self, X, y, metric_arr):
        mestimator = MEstimator()
        huber = Huber()
        loc = mestimator.loc_estimator(metric_arr)
        return huber.m_weights(metric_arr - loc)




if __name__ == '__main__':
    X_train, y_train, X_test, y_test = simulation_setup(n_i=1000, n_o=200, n_t=1000, p=8,
                                                        sigma_e=0.25)
    mod_stbp = ModifiedStraitifiedBootstrap()
    mod_stbp.fit(X_train, y_train, n_bootstrap=20, metric='leverage')

    print("modified strafified bootstrap accuracy: ", mod_stbp.score(X_test, y_test))





