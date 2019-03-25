import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('MacOSX')
from rblr.influence_function_bootstrap import IFB
from rblr.classical_bootstrap import ClassicalBootstrap
from sklearn.linear_model import LogisticRegression
from rblr.stratified_bootstrap import StratifiedBootstrap
from rblr.simulation_setup import simulation_setup
matplotlib.rcParams['font.size'] = 10

p = 10
n_strata_arr = np.arange(2, 16, 2, dtype=int)

# lambda_ = np.arange(0.05, 0.55, 0.05)
n_i = 1000
# n_o_arr = (n_i * lambda_).astype(int)

score_strat_arr = []
# score_boot_arr = []

# for n_o in n_o_arr:
for n_strata in n_strata_arr:
    score_strat = []
    # score_boot = []
    for i in range(10):
        X_train, y_train, X_test, y_test = simulation_setup(n_i=n_i, n_o=200, n_t=n_i, p=p)

        # stratified
        strat = StratifiedBootstrap()
        strat.fit(X_train, y_train, n_bootstrap=5, n_strata=n_strata, fast=True)
        score_strat.append(strat.score(X_test, y_test))

        # Bootstrap
        boot = ClassicalBootstrap()
        boot.fit(X_train, y_train, n_bootstrap=5)
        # score_boot.append(boot.score(X_test, y_test))

    score_strat_arr.append(score_strat)
    # score_boot_arr.append(score_boot)

score_mean_strat = np.mean(score_strat_arr, axis=1)
# score_mean_boot = np.mean(score_boot_arr, axis=1)

score_std_strat = np.std(score_strat_arr, axis=1, ddof=1)
# score_std_boot = np.std(score_boot_arr, axis=1, ddof=1)

f1 = plt.figure(1)
plt.errorbar(n_strata_arr, score_mean_strat, yerr=score_std_strat,
             capsize=2, fmt='-o', label='Stratified Bootstrap')
# plt.errorbar(lambda_, score_mean_boot, yerr=score_std_boot,
#              capsize=2, fmt='-o', label='Classical Bootstrap')
plt.xlabel('Number of strata')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.show()