import numpy as np
from rblr.bisquare import Bisquare
import time
from rblr.simulation_setup import simulation_setup

class MEstimator:
    def __init__(self, clipping=4.685):
        """
        :param clipping: clipping constant for bisquare function

        """
        self.biweight = Bisquare(clipping=clipping)

    def scale_estimator(self, arr, axis=0, maxiter=50, tor=0.001):
        """
        calculate the M-estimator of scale along a given axis iteratively.

        :param arr: array-like, 2-D or 1-D, input array
        :param axis: int (0,1), along which axis the M-estimator will be computed
        :param maxiter: int, the maximal number of iterations
        :param tor: float, error tolerance level, in interval (0,1)
        :return: ndarray
        """
        def unit_scale_estimator(element, maxiter, tor):
            element = np.array(element, dtype=np.float)
            element = element - np.median(element)
            # compute MADN as initial guess of M-estimator of scale
            scale = 1.483 * np.median(np.abs(element))

            # iterative computation of scale estimator
            for i in range(maxiter):
                # update weights and scale
                # $ sigma_{k+1} = sqrt( \frac{1}{n \delta} \sum_{i=1}^{n} w_{k,i} x_i^2 ) $, here \delta = 0.42
                weights = self.biweight.m_weights(element/scale)
                scale_new = np.sqrt(np.dot(weights, np.square(element)) / (len(element) * 0.42))
                # print("scale_new: %f, scale_old: %f"%(scale_new, scale))
                # print('diff_scale: ', np.abs(scale_new/scale - 1))
                if np.abs(scale_new/scale - 1) < tor or \
                        np.abs(scale_new/scale - 1) > 0.95:
                    break
                scale = scale_new
            return scale

        # vectorized function
        return np.apply_along_axis(unit_scale_estimator, axis, arr, maxiter=maxiter, tor=tor)

    def loc_estimator(self, arr, axis=0, maxiter=50, tor=0.001):
        """
        calculate the M-estimator of location along a given axis

        :param arr: array-like, 1-D or 2-D input array
        :param axis: int, 0 or 1, along which axis the estimator will be calculated
        :param maxiter: int, maximal number of iterations
        :param tor: float in interval (0,1) , error tolerance level
        :return: ndarray
        """
        def unit_loc_estimator(element, maxiter, tor):
            element = np.array(element, dtype=np.float)
            # initial estimator of scale
            sigma = 1.483 * np.median(np.abs(element - np.median(element)))
            # initial location estimator
            mu = np.median(element)

            for i in range(maxiter):
                weights = self.biweight.m_weights((element - mu)/sigma)
                mu_new = np.dot(weights, element) / np.sum(weights)

                if np.abs(mu_new - mu) < tor * sigma:
                    break
                mu = mu_new
            return mu

        # vectorized function
        return np.apply_along_axis(unit_loc_estimator, axis, arr, maxiter=maxiter, tor=tor)

    def fit(self, X, axis=0, maxiter=50, tor=0.001):
        loc = self.loc_estimator(X, axis=axis, maxiter=maxiter, tor=tor)
        scale = self.scale_estimator(X, axis=axis, maxiter=maxiter, tor=tor)
        return loc, scale


if __name__ == '__main__':
    m_est = MEstimator()
    X_train, y_train, X_test, y_test = simulation_setup(n_i=1000, n_o=200, n_t=1000, p=10,
                                                        sigma_e=0.25)


    t1 = time.time()
    m_estimated_scale = m_est.scale_estimator(X_train, maxiter=50)
    m_estimated_loc = m_est.loc_estimator(X_train, maxiter=50)
    standard_deviation = np.std(X_train, ddof=1, axis=0)
    median = np.median(X_train, axis=0)
    print('consumed time: %.5f s' % (time.time() - t1))