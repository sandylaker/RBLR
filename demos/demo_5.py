import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('MacOSX')
from rblr.influence_function_bootstrap import IFB
from rblr.classical_bootstrap import ClassicalBootstrap
from sklearn.linear_model import LogisticRegression
from rblr.preprocessing import Preprocessor
from rblr.simulation_setup import simulation_setup
matplotlib.rcParams['font.size'] = 10


p = 2

lambda_ = np.arange(0.05, 0.55, 0.05)
n_i = 1000
n_o_arr = (n_i * lambda_).astype(int)

score_ppifb_arr = []
score_ifb_arr = []
score_boot_arr = []

for n_o in n_o_arr:
    score_ppifb = []
    score_ifb = []
    score_boot = []
    for i in range(10):
        X_train, y_train, X_test, y_test = simulation_setup(n_i=n_i, n_o=n_o, n_t=n_i, p=p)

        # preprocessed IFB
        preprocessor = Preprocessor()
        X_train_preprocessed, y_train_preprocessed = preprocessor.fit_transform(X_train, y_train)

        ifb = IFB(gamma=8)
        ifb.fit(X_train_preprocessed, y_train_preprocessed, n_bootstrap=5,
                quantile_factor=np.random.uniform(0.6, 0.95))
        score_ppifb.append(ifb.score(X_test, y_test))

        # IFB
        ifb = IFB()
        ifb.fit(X_train, y_train, n_bootstrap=5, quantile_factor=0.8)
        score_ifb.append(ifb.score(X_test, y_test))

        # Bootstrap
        boot = ClassicalBootstrap()
        boot.fit(X_train, y_train, n_bootstrap=5)
        score_boot.append(boot.score(X_test, y_test))

    score_ppifb_arr.append(score_ppifb)
    score_ifb_arr.append(score_ifb)
    score_boot_arr.append(score_boot)

score_mean_ppifb = np.mean(score_ppifb_arr, axis=1)
score_mean_ifb = np.mean(score_ifb_arr, axis=1)
score_mean_boot = np.mean(score_boot_arr, axis=1)

score_std_ppifb = np.std(score_ppifb_arr, axis=1, ddof=1)
score_std_ifb = np.std(score_ifb_arr, axis=1, ddof=1)
score_std_boot = np.std(score_boot_arr, axis=1, ddof=1)

f1 = plt.figure(1)
plt.errorbar(lambda_, score_mean_ppifb, yerr=score_std_ppifb,
             capsize=2, fmt='-o', label='Preprocessed IFB')
plt.errorbar(lambda_, score_mean_ifb, yerr=score_std_ifb,
             capsize=2, fmt='-o', label='IFB')
plt.errorbar(lambda_, score_mean_boot, yerr=score_std_boot,
             capsize=2, fmt='-o', label='Classical Bootstrap')
plt.xlabel(r'$\lambda = N_{out}/N$')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.show()

