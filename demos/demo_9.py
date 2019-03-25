import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('MacOSX')
from rblr.influence_function_bootstrap import IFB
from rblr.classical_bootstrap import ClassicalBootstrap
from sklearn.linear_model import LogisticRegression
from rblr.stratified_bootstrap import StratifiedBootstrap
from rblr import simulation_setup, imbalanced_simulation, ModifiedStraitifiedBootstrap
matplotlib.rcParams['font.size'] = 10

p = 10
n_strata_arr = np.arange(2, 10, 1, dtype=int)

n_i = 1000

score_mod_arr = []

for n_strata in n_strata_arr:
    score_mod = []
    for i in range(10):
        X_train, y_train, X_test, y_test = simulation_setup(n_i=n_i, n_o=200, n_t=n_i, p=p)

        # stratified
        mod_strat = ModifiedStraitifiedBootstrap()
        mod_strat.fit(X_train, y_train, n_bootstrap=5, n_strata=n_strata)
        score_mod.append(mod_strat.score(X_test, y_test))
    score_mod_arr.append(score_mod)

score_mean_mod = np.mean(score_mod_arr, axis=1)

score_std_mod = np.std(score_mod_arr, axis=1, ddof=1)

f1 = plt.figure(1)
plt.errorbar(n_strata_arr, score_mean_mod, score_std_mod,
             capsize=2, fmt='-o', label='Modified Stratified \n Bootstrap')
plt.xlabel('Number of Strata')
plt.ylabel('Accuracy')
plt.ylim((0.5, 1))
plt.legend(loc='best')
plt.show()