import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from rblr.influence_function_bootstrap import IFB
from rblr.classical_bootstrap import ClassicalBootstrap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE
from rblr.stratified_bootstrap import StratifiedBootstrap
from rblr import simulation_setup, imbalanced_simulation, ModifiedStraitifiedBootstrap, Preprocessor
matplotlib.rcParams['font.size'] = 10

df = pd.read_csv('../test_dataset/breast_cancer.csv')
df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})

X = df.drop(['diagnosis'], axis=1)
y = df['diagnosis']

smote = SMOTE(sampling_strategy={0: 600, 1: 600}, k_neighbors=6)
X, y = smote.fit_resample(X, y)

min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


def contaminate(X_train, y_train, contamination=0.2, scale=10, label_percentage=0.5):
    if ((np.max(X_train, axis=0) - np.min(X_train, axis=0)) > 1.1).any():
        print(np.max(X_train, axis=0) - np.min(X_train, axis=0))
        raise ValueError("the input matrix is not min-max scaled")
    if label_percentage < 0 or label_percentage > 1:
        raise ValueError("label percentage can only be between 0 and 1")
    np.random.seed()
    n_out = int(X_train.shape[0] * contamination)
    n_out_0 = int(n_out * label_percentage)
    n_out_1 = n_out - n_out_0

    # generate outliers
    X_cat_0 = X_train[y_train == 0]
    X_out_0 = scale * X_cat_0[np.random.choice(X_cat_0.shape[0], n_out_0)] + \
              0.1 * scale * np.random.randn(n_out_0, X_cat_0.shape[1])
    # reverse label of 0-class into 1
    y_out_0 = np.ones(n_out_0, dtype=int)

    X_cat_1 = X_train[y_train == 1]
    X_out_1 = scale * X_cat_1[np.random.choice(X_cat_1.shape[0], n_out_1)] + \
              0.1 * scale * np.random.rand(n_out_1, X_cat_1.shape[1])
    # reverse label of 1-class into 0
    y_out_1 = np.zeros(n_out_1, dtype=int)

    # concatenate X and y
    X_train = np.concatenate((X_train, X_out_0, X_out_1), axis=0)
    y_train = np.concatenate((y_train, y_out_0, y_out_1), axis=0)
    return X_train, y_train


# on clean data
lr = LogisticRegression(solver='lbfgs')
lr.fit(X_train, y_train)
print('LR on clean data: ', lr.score(X_test, y_test))

boot = ClassicalBootstrap()
boot.fit(X_train, y_train, n_bootstrap=5)
print('Bootstrap on clean data', boot.score(X_test, y_test))

ifb = IFB(gamma=5)
ifb.fit(X_train, y_train, quantile_factor=0.85)
print('IFB on clean data', ifb.score(X_test, y_test))

preprocessor = Preprocessor()
X_prep, y_prep = preprocessor.fit_transform(X_train, y_train)
ifb = IFB(gamma=5)
ifb.fit(X_train, y_train, quantile_factor=0.95)
print('PPIFB on clean data', ifb.score(X_test, y_test))

strat = StratifiedBootstrap()
strat.fit(X_train, y_train, n_bootstrap=5, n_strata=3, fast=True)
print('Stratified Bootstrap on clean data', strat.score(X_test, y_test))

mod = ModifiedStraitifiedBootstrap()
mod.fit(X_train, y_train, n_bootstrap=5, n_strata=3)
print('Modified Stratified Bootstrap on clean data', mod.score(X_test, y_test))

# contamination = np.arange(0.05, 0.55, 0.05)
# scale = 10
#
# score_lr_arr = []
# score_boot_arr = []
# score_ifb_arr = []
# score_pp_arr = []
# score_strat_arr = []
# score_mod_arr = []
#
# for contam_level in contamination:
#     score_lr = []
#     score_boot = []
#     score_ifb = []
#     score_pp = []
#     score_strat = []
#     score_mod = []
#
#     print('============================')
#     print('contamination level: %.3f' % contam_level)
#     print('============================')
#     for i in range(10):
#         print('round %d' % i)
#         X_train_conta, y_train_conta = contaminate(X_train, y_train, contam_level, scale=scale)
#
#         # Classical LR
#         lr = LogisticRegression(solver='lbfgs', max_iter=500)
#         lr.fit(X_train_conta, y_train_conta)
#         score_lr.append(lr.score(X_test, y_test))
#
#         # Classical Bootstrap
#         boot = ClassicalBootstrap(max_iter=500)
#         boot.fit(X_train_conta, y_train_conta, n_bootstrap=5)
#         score_boot.append(boot.score(X_test, y_test))
#
#         # IFB
#         ifb = IFB(gamma=5)
#         ifb.fit(X_train_conta, y_train_conta, n_bootstrap=5, max_iter=500, quantile_factor=0.85)
#         score_ifb.append(ifb.score(X_test, y_test))
#
#         # PPIFB
#         preprocessor = Preprocessor()
#         X_prep, y_prep = preprocessor.fit_transform(X_train_conta, y_train_conta)
#         ifb = IFB(gamma=5)
#         ifb.fit(X_prep, y_prep, n_bootstrap=5, max_iter=500, quantile_factor=0.95)
#         score_pp.append(ifb.score(X_test, y_test))
#
#         # Stratified Bootstrap
#         strat = StratifiedBootstrap(max_iter=500)
#         strat.fit(X_train_conta, y_train_conta, n_bootstrap=5, n_strata=3, fast=True)
#         score_strat.append(strat.score(X_test, y_test))
#
#         # Modified Stratified Bootstrap
#         mod = ModifiedStraitifiedBootstrap(max_iter=500)
#         mod.fit(X_train_conta, y_train_conta, n_bootstrap=5, n_strata=3)
#         score_mod.append(mod.score(X_test, y_test))
#
#     score_lr_arr.append(score_lr)
#     score_boot_arr.append(score_boot)
#     score_ifb_arr.append(score_ifb)
#     score_pp_arr.append(score_pp)
#     score_strat_arr.append(score_strat)
#     score_mod_arr.append(score_mod)
#
# score_lr_mean = np.mean(score_lr_arr, axis=1)
# score_lr_std = np.std(score_lr_arr, axis=1, ddof=1)
#
# score_boot_mean = np.mean(score_boot_arr, axis=1)
# score_boot_std = np.std(score_boot_arr, axis=1, ddof=1)
#
# score_ifb_mean = np.mean(score_ifb_arr, axis=1)
# score_ifb_std = np.std(score_ifb_arr, axis=1, ddof=1)
#
# score_pp_mean = np.mean(score_pp_arr, axis=1)
# score_pp_std = np.std(score_pp_arr, axis=1, ddof=1)
#
# score_strat_mean = np.mean(score_strat_arr, axis=1)
# score_strat_std = np.std(score_strat_arr, axis=1, ddof=1)
#
# score_mod_mean = np.mean(score_mod_arr, axis=1)
# score_mod_std = np.std(score_mod_arr, axis=1, ddof=1)
#
# f1 = plt.figure(1)
# plt.errorbar(contamination, score_lr_mean, score_lr_std,
#              capsize=2, fmt='-o', label='Classical LR')
# plt.errorbar(contamination, score_boot_mean, score_boot_std,
#              capsize=2, fmt='-o', label='Classical Bootstrap')
# plt.errorbar(contamination, score_ifb_mean, score_ifb_std,
#              capsize=2, fmt='-o', label='IFB')
# plt.errorbar(contamination, score_pp_mean, score_pp_std,
#              capsize=2, fmt='-o', label='Preprocessed IFB')
# plt.errorbar(contamination, score_strat_mean, score_strat_std,
#              capsize=2, fmt='-o', label='Stratified Bootstrap')
# plt.errorbar(contamination, score_mod_mean, score_mod_std,
#              capsize=2, fmt='-o', label='Modified Stratified \n Bootstrap')
# plt.xlabel(r'$\lambda = N_{out}/ N$')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower left')
# plt.show()


contam_level = 0.3
scale = 10
X_conta, y_conta = contaminate(X_train, y_train, contam_level, scale=10)

lr = LogisticRegression(solver='lbfgs', max_iter=500)
lr.fit(X_conta, y_conta)
y_score_lr = lr.predict_proba(X_test)[:, -1]
auc_lr = roc_auc_score(y_test, y_score_lr)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_score_lr)

boot = ClassicalBootstrap(max_iter=500)
boot.fit(X_conta, y_conta, n_bootstrap=5)
y_score_boot = boot.predict_proba(X_test)[:, -1]
auc_boot = roc_auc_score(y_test, y_score_boot)
fpr_boot, tpr_boot, _ = roc_curve(y_test, y_score_boot)

ifb = IFB(gamma=5)
ifb.fit(X_conta, y_conta, n_bootstrap=5, quantile_factor=0.85, max_iter=500)
y_score_ifb = ifb.predict_proba(X_test)[:, -1]
auc_ifb = roc_auc_score(y_test, y_score_ifb)
fpr_ifb, tpr_ifb, _ = roc_curve(y_test, y_score_ifb)

preprocessor = Preprocessor()
X_prep, y_prep = preprocessor.fit_transform(X_conta, y_conta)
ifb = IFB(gamma=5)
ifb.fit(X_prep, y_prep, n_bootstrap=5, quantile_factor=0.85, max_iter=500)
y_score_pp = ifb.predict_proba(X_test)[:, -1]
auc_pp = roc_auc_score(y_test, y_score_pp)
fpr_pp, tpr_pp, _ = roc_curve(y_test, y_score_pp)

strat = StratifiedBootstrap(max_iter=500)
strat.fit(X_conta, y_conta, n_bootstrap=5, n_strata=3, fast=True)
y_score_strat = strat.predict_proba(X_test)[:, -1]
auc_strat = roc_auc_score(y_test, y_score_strat)
fpr_strat, tpr_strat, _ = roc_curve(y_test, y_score_strat)

mod = ModifiedStraitifiedBootstrap(max_iter=500)
mod.fit(X_conta, y_conta, n_bootstrap=5, n_strata=3)
y_score_mod = mod.predict_proba(X_test)[:, -1]
auc_mod = roc_auc_score(y_test, y_score_mod)
fpr_mod, tpr_mod, _ =roc_curve(y_test, y_score_mod)

f1 = plt.figure(1)
plt.plot(fpr_lr, tpr_lr, label='Classical LR')
plt.plot(fpr_boot, tpr_boot, label='Classical Bootstrap')
plt.plot(fpr_ifb, tpr_ifb, label='Classical IFB')
plt.plot(fpr_pp, tpr_pp, label='Preprocessed IFB')
plt.plot(fpr_strat, tpr_strat, label='Stratified Bootstrap')
plt.plot(fpr_mod, tpr_mod, label='Modified Stratified \n Bootstrap')
plt.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10), '--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.xlim((-0.01, 1.01))
plt.ylim((-0.01, 1.01))
plt.legend(loc=(1.01, 0.0))
plt.show()




