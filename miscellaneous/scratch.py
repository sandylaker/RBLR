import numpy as np
from sklearn.linear_model import  LogisticRegression
from simulation_setup import simulation_setup
from influence_function_bootstrap import IFB
from sklearn.metrics import classification_report
from preprocessing import Preprocessor

data_train, data_test, beta_actual = simulation_setup(n_i=1000, n_o=500, n_t=1000, p=2, sigma_e=0.25)
X_train, y_train = data_train[:, :-1], data_train[:, -1].astype(int)
X_test, y_test = data_test[:, :-1], data_test[:, -1].astype(int)

clf_ifb = IFB()
clf_ifb.fit(X_train, y_train, quantile_factor=0.7)
beta_ifb = clf_ifb.beta

clf_lr = LogisticRegression(fit_intercept=False, solver='lbfgs')
clf_lr.fit(X_train, y_train)

y_predict_ifb = clf_ifb.predict(X_test)
y_predict_lr = clf_lr.predict(X_test)

print("Classical LR: \n", classification_report(y_test, y_predict_lr))
print("IFB LR: \n", classification_report(y_test, y_predict_ifb))

preprocessor = Preprocessor()
X_train_in, y_train_in = preprocessor.preprocessing(X_train, y_train)
clf_pre_ifb = IFB()
clf_pre_ifb.fit(X_train_in, y_train_in, quantile_factor=0.7)
beta_pre_ifb = clf_pre_ifb.beta

y_predict_pre_ifb = clf_pre_ifb.predict(X_test)
print("Preprocessed IFB LR: \n", classification_report(y_test, y_predict_pre_ifb))

