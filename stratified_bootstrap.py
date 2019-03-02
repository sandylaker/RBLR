import numpy as np
from wmle import WMLE
from simulation_setup import simulation_setup
from sklearn.linear_model import LogisticRegression
from classical_bootstrap import ClassicalBootstrap
from util_funcs import deviance_residual
import time

class StratifiedBootstrap:

    def __init__(self, clipping=1.345, fit_intercept=True, warm_start=True):
        self.wmle = WMLE(clipping=clipping, fit_intercept=fit_intercept, warm_start=warm_start)
        self.fit_intercept = self.wmle.fit_intercept
        self.beta = None
        self.intercept_ = None
        self.coef_ = None

    def stratify(self, X_y, n_strata=10, sort='residual'):
        if sort == 'leverage':
            X_ = X_y[:, :-1]
            metric_arr = self.wmle.leverage(X_)
        elif sort == 'residual':
            # in fit function, X and y are concatenated, so split the input array
            X_, y_ = X_y[:, :-1], X_y[:, -1].astype(int)
            lr = LogisticRegression(fit_intercept=self.fit_intercept, solver='lbfgs')
            lr.fit(X_, y_)
            log_prob = lr.predict_log_proba(X_)
            metric_arr = deviance_residual(y_, log_prob)
        else:
            raise ValueError("set sort either to 'residual' or 'leverage' ")
        stratas = []
        counts = []
        if not X_y.shape[0] % n_strata == 0:
            q = np.linspace(0, 1, n_strata + 1)
            bins = np.quantile(metric_arr, q)
            for i in range(len(bins) - 2):
                stratas.append(X_y[(metric_arr >= bins[i]) & (metric_arr < bins[i+1])])
                counts.append(len(X_y[-1]))
            # append the stratum from last bin
            stratas.append(X_y[(metric_arr >= bins[-2]) & (metric_arr <= bins[-1])])
            counts.append(len(X_y[-1]))
        else:
            order = np.argsort(metric_arr)
            X_y = X_y[order]
            stratas = np.split(X_y, n_strata)
            counts = [len(stratas[0])] * len(stratas)
        return stratas, np.array(counts)

    def get_n_candidates(self, n_samples_each_bootstrap, counts):
        """
        calculate the numbers of candidates in each stratum indicating how many candidates in
        each group will finally enter the bootstrap sample

        :param n_samples_each_bootstrap: number of observations in each bootstrap sample

        :param counts: array_like, counts of sorted residuals in each bin, the candidates will be
        drawn from each stratum with the proportion calculated from the counts array.

        :return: ndarray, array of number of candidates from each stratum
        """
        n_canditates = np.zeros(len(counts), dtype=int)
        n_canditates[1:] = np.floor(
            n_samples_each_bootstrap * counts / np.sum(counts)).astype(int)[1:]
        n_canditates[0] = n_samples_each_bootstrap - np.sum(n_canditates)
        return n_canditates

    def resample(self, stratas, n_bootstrap, n_candidates):
        bootstrap_samples = []
        stack = []
        for i, stratum in enumerate(stratas):
            index_range = np.arange(stratum.shape[0])
            index_arr = np.random.choice(index_range, (n_bootstrap, n_candidates[i]))
            samples_of_stratum = stratum[index_arr]
            stack.append(samples_of_stratum)
        for i in range(n_bootstrap):
            temp = []
            for samples_of_stratum in stack:
                temp.append(samples_of_stratum[i])
            bootstrap_samples.append(np.concatenate(temp, axis=0))
        return np.array(bootstrap_samples)

    def fit(self, X, y, n_bootstrap=20,
            n_strata=10,
            n_samples_each_bootstrap=None,
            sort='residual',
            verbose=False):

        y = np.asarray(y).astype(int)
        if y.ndim == 1:
            y = y[:, np.newaxis]
        # concatenate X and y horizontally for convenience
        X_concat = np.concatenate((X, y), axis=1)

        if n_samples_each_bootstrap is None:
            n_samples_each_bootstrap = X_concat.shape[0]

        # stratify the input
        stratas, counts = self.stratify(X_concat, n_strata=n_strata, sort=sort)
        n_candidates = self.get_n_candidates(n_samples_each_bootstrap, counts)

        # resampling
        bootstrap_samples = self.resample(stratas, n_bootstrap, n_candidates)
        self.beta = []
        for i, X_concat_b in enumerate(bootstrap_samples):
            if verbose:
                print("-----fit bootstrap sample No.{}-----".format(i))
            # split X and y
            X_b = X_concat_b[:, :-1]
            y_b = X_concat_b[:, -1]
            self.wmle.fit(X_b, y_b)
            self.beta.append(self.wmle.beta)
        self.beta = np.mean(self.beta, axis=0)
        # normalize the average of coefficients
        self.beta = self.beta / np.linalg.norm(self.beta)
        if self.wmle.fit_intercept:
            self.intercept_ = self.beta[1:]
        self.coef_ = self.beta[self.wmle.fit_intercept:]
        return self

    def predict(self, X_predict, prob_threshold=0.5):
        self.wmle.set_beta(beta=self.beta)
        return self.wmle.predict(X_predict, prob_threshold)

    def score(self, X_test, y_test, prob_threshold=0.5):
        y_test = np.array(y_test, dtype=int)
        y_predict = self.predict(X_test, prob_threshold)
        accuracy = np.mean(y_predict == y_test)
        return accuracy




if __name__ == '__main__':
    # stratas = [np.arange(12).reshape(3, 4),
    #            np.arange(0, 16).reshape(4, 4),
    #            np.arange(0, 20).reshape(5, 4)]
    # n_candidates = np.arange(3, 6)
    # stbp = StratifiedBootstrap()
    # resamples = stbp.resample(stratas, 3, n_candidates)

    X_train, y_train, X_test, y_test = simulation_setup(n_i=1000, n_o=300, n_t=1000, p=8,
                                                          sigma_e=0.25)


    stbp =StratifiedBootstrap(warm_start=True, fit_intercept=True)
    t1 = time.time()
    stbp.fit(X_train, y_train, n_bootstrap=10, n_strata=10, sort='residual', verbose=True)
    print("Consumed time of Stratified Bootstrap fit %.2f s" % (time.time() - t1))

    lr = LogisticRegression(fit_intercept=True, solver='lbfgs')
    lr.fit(X_train, y_train)

    clbp = ClassicalBootstrap()
    clbp.fit(X_train, y_train, n_bootstrap=10)

    print("Classical LR accuracy: ", lr.score(X_test, y_test))
    print("Classical Bootstrap accuracy: ", clbp.score(X_test, y_test))
    print("Stratified Bootstrap accuracy: ", stbp.score(X_test, y_test))



