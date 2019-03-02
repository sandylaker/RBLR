import numpy as np
from sklearn.linear_model import  LogisticRegression
from simulation_setup import simulation_setup
from influence_function_bootstrap import IFB
from sklearn.metrics import classification_report, precision_recall_fscore_support
from preprocessing import Preprocessor

import matplotlib.pyplot as plt


n_i = 1000
n_o = 200
n_t = 1000
p = 20
sigma_e = 0.25
quantile_factor = np.arange(0.5, 1.0, 0.05)
X_train, y_train, X_test, y_test = simulation_setup(n_i=n_i, n_o=n_o, n_t=n_t, p=p,
                                                      sigma_e=sigma_e)
# print("Simulation setup: inliers: {}; outliers: {}; dimensions: {}".format(n_i, n_o, n_t, p, sigma_e))


# def get_metrics_matrix(quantile_factor):
#     metrics_matrix = np.empty((len(quantile_factor), 3, 4))
#     for i, q in enumerate(quantile_factor):
#         unit_matrix = np.zeros((3, 4))
#         clf_ifb = IFB()
#         clf_ifb.fit(X_train, y_train, quantile_factor=q)
#         beta_ifb = clf_ifb.beta
#
#         clf_lr = LogisticRegression(fit_intercept=False, solver='lbfgs')
#         clf_lr.fit(X_train, y_train)
#
#         y_predict_ifb = clf_ifb.predict(X_test)
#         y_predict_lr = clf_lr.predict(X_test)
#
#         # print("Classical LR: \n", classification_report(y_test, y_predict_lr))
#         # print("IFB LR: \n", classification_report(y_test, y_predict_ifb))
#
#         preprocessor = Preprocessor()
#         X_train_in, y_train_in = preprocessor.fit_transform(X_train, y_train)
#         clf_pre_ifb = IFB()
#         clf_pre_ifb.fit(X_train_in, y_train_in, quantile_factor=q)
#         beta_pre_ifb = clf_pre_ifb.beta
#
#         y_predict_pre_ifb = clf_pre_ifb.predict(X_test)
#         # print("Preprocessed IFB LR: \n", classification_report(y_test, y_predict_pre_ifb))
#         unit_matrix[0] = precision_recall_fscore_support(y_test, y_predict_lr, average='macro')
#         unit_matrix[1] = precision_recall_fscore_support(y_test, y_predict_ifb, average='macro')
#         unit_matrix[2] = precision_recall_fscore_support(y_test, y_predict_pre_ifb, average='macro')
#         metrics_matrix[i] = unit_matrix
#     return np.array(metrics_matrix)

def get_lr_metrics(quantile_factor=quantile_factor):
    ret = []
    for q in quantile_factor:
        clf_lr = LogisticRegression(fit_intercept=True, solver='lbfgs')
        clf_lr.fit(X_train, y_train)
        y_predict_lr = clf_lr.predict(X_test)
        ret.append(precision_recall_fscore_support(y_test, y_predict_lr, average='macro'))
    return np.array(ret)

def get_ifb_metrics(quantile_factor=quantile_factor):
    ret = []
    for q in quantile_factor:
        clf_ifb = IFB(fit_intercept=True)
        clf_ifb.fit(X_train, y_train, quantile_factor=q)
        y_predict_ifb = clf_ifb.predict(X_test)
        ret.append(precision_recall_fscore_support(y_test, y_predict_ifb, average='macro'))
    return np.array(ret)

def get_ppifb_metrics(quantile_factor=quantile_factor):
    ret = []
    preprocessor = Preprocessor()
    X_train_in, y_train_in = preprocessor.fit_transform(X_train, y_train)
    for q in quantile_factor:
        clf_pre_ifb = IFB(fit_intercept=True)
        clf_pre_ifb.fit(X_train_in, y_train_in, quantile_factor=q)
        y_predict_pre_ifb = clf_pre_ifb.predict(X_test)
        ret.append(precision_recall_fscore_support(y_test, y_predict_pre_ifb, average='macro'))
    return np.array(ret)


lr_metrics = get_lr_metrics(quantile_factor=quantile_factor)
ifb_metrics = get_ifb_metrics(quantile_factor)
ppifb_metrics = get_ppifb_metrics(quantile_factor)

# Notice: the classical LR is independent of quantile factor, so the metrics of it are constants
# no matter how the quantile factor changes. We plot it as a comparison

# plot precision
plt.figure(1)
plt.plot(quantile_factor, lr_metrics[:, 0], label='Classical LR')
plt.plot(quantile_factor, ifb_metrics[:, 0], label='IFB')
plt.plot(quantile_factor, ppifb_metrics[:, 0], label='Preprocessed IFB')
plt.xlabel('Quantile Factor')
plt.ylabel('Precision')
plt.title('Precision')
plt.legend()

# plot recall
plt.figure(2)
plt.plot(quantile_factor, lr_metrics[:, 1], label='Classical LR')
plt.plot(quantile_factor, ifb_metrics[:, 1], label='IFB')
plt.plot(quantile_factor, ppifb_metrics[:, 1], label='Preprocessed IFB')
plt.xlabel('Quantile Factor')
plt.ylabel('Recall')
plt.title('Recall')
plt.legend()

# plot F1 score
plt.figure(3)
plt.plot(quantile_factor, lr_metrics[:, 2], label='Classical LR')
plt.plot(quantile_factor, ifb_metrics[:, 2], label='IFB')
plt.plot(quantile_factor, ppifb_metrics[:, 2], label='Preprocessed IFB')
plt.xlabel('Quantile Factor')
plt.ylabel('F1 Score')
plt.title('F1 Score')
plt.legend()
plt.show()








