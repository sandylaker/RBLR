import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('MacOSX')
matplotlib.rcParams['font.size'] = 10
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from rblr import simulation_setup

X_train, y_train, X_test, y_test = simulation_setup(1000, 0, 1000, 10)

lr = LogisticRegression(solver='lbfgs')
lr.fit(X_train, y_train)

y_score = lr.predict_proba(X_test)[:, -1]
auc = roc_auc_score(y_test, y_score)

fpr, tpr, thres = roc_curve(y_test, y_score)

f1 = plt.figure(1)
plt.plot(fpr, tpr, linewidth=2)
plt.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10), '--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.xlim((-0.01, 1.01))
plt.ylim((0, 1))
plt.show()