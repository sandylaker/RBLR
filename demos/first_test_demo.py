from rblr import *
from multiprocessing import Pool
import numpy as np
import time

def test_accuracy(x):
    if x > 0 :
        X_train, y_train, X_test, y_test = simulation_setup(n_i=1000, n_o=200, n_t=1000, p=8,
                                                                  sigma_e=0.25)


        stbp =StratifiedBootstrap(warm_start=True, fit_intercept=True)
        t1 = time.time()
        stbp.fit(X_train, y_train, n_bootstrap=5, n_strata=10, sort='residual', verbose=False)
        print("Consumed time of Stratified Bootstrap fit %.2f s" % (time.time() - t1))

        lr = LogisticRegression(fit_intercept=True, solver='lbfgs')
        lr.fit(X_train, y_train)

        clbp = ClassicalBootstrap()
        clbp.fit(X_train, y_train, n_bootstrap=5)

        # print("Classical LR accuracy: ", lr.score(X_test, y_test))
        # print("Classical Bootstrap accuracy: ", clbp.score(X_test, y_test))
        # print("Stratified Bootstrap accuracy: ", stbp.score(X_test, y_test))
        return [lr.score(X_test, y_test), clbp.score(X_test, y_test), stbp.score(X_test, y_test)]

# pool = Pool(processes=3)
# t1 = time.time()
# res = pool.map(test_accuracy, range(1, 3))
# print("consumed time: %.2f s" % (time.time() - t1))


if __name__ == '__main__':
    def func(x, y):
        return x, y, x + y
    ys = np.arange(3)
    p1 = Pool(processes=8)
    res = p1.starmap(func, enumerate(ys))
    print(res)