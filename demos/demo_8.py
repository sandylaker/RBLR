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

lambda_ = np.arange(0.05, 0.55, 0.05)
n_i = 1000
n_o_arr = (n_i * lambda_).astype(int)

score_mod_arr = []
score_boot_arr = []

for n_o in n_o_arr:
    score_mod = []
    score_boot = []
    for i in range(10):
        X_train, y_train, X_test, y_test = simulation_setup(n_i=n_i, n_o=n_o, n_t=n_i, p=p)


        modified_boot = ModifiedStraitifiedBootstrap()
        modified_boot.fit(X_train, y_train, n_strata=5)
        score_mod.append(modified_boot.score(X_test, y_test))

        # Bootstrap
        boot = ClassicalBootstrap()
        boot.fit(X_train, y_train, n_bootstrap=5)
        score_boot.append(boot.score(X_test, y_test))

    score_mod_arr.append(score_mod)
    score_boot_arr.append(score_boot)

score_mean_mod = np.mean(score_mod_arr, axis=1)
score_mean_boot = np.mean(score_boot_arr, axis=1)

score_std_mod = np.std(score_mod_arr, axis=1, ddof=1)
score_std_boot = np.std(score_boot_arr, axis=1, ddof=1)

f1 = plt.figure(1)
plt.errorbar(lambda_, score_mean_mod, yerr=score_std_mod,
             capsize=2, fmt='-o', label='Modified Stratified \n Bootstrap')

plt.errorbar(lambda_, score_mean_boot, yerr=score_std_boot,
             capsize=2, fmt='-o', label='Classical Bootstrap')
plt.xlabel(r'$\lambda = N_{out}/N$')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.show()