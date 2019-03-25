import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('MacOSX')
from rblr.influence_function_bootstrap import IFB
from rblr.classical_bootstrap import ClassicalBootstrap
from sklearn.linear_model import LogisticRegression
from rblr.stratified_bootstrap import StratifiedBootstrap
from rblr import simulation_setup, imbalanced_simulation
matplotlib.rcParams['font.size'] = 10


# X_train, y_train, X_test, y_test = imbalanced_simulation(0.5, n_i=2000, n_o=200, n_t=1000, p=5)
#
# lr = LogisticRegression(solver='lbfgs')
# lr.fit(X_train, y_train)
# print("LR: ", lr.score(X_test, y_test))
#
# ifb = IFB()
# ifb.fit(X_train, y_train)
# print("IFB: ", ifb.score(X_test, y_test))

X_train, y_train, X_test, y_test = simulation_setup(n_i=100, n_o=20, n_t=100, p=2)
X_in, X_out = X_train[: 100], X_train[100:]
y_in, y_out = y_train[: 100], y_train[100:]

lr = LogisticRegression(solver='lbfgs')
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))

s = 8
f1 = plt.figure(1)
X_in_0, X_in_1 = X_in[y_in == 0], X_in[y_in == 1]
X_out_0, X_out_1 = X_out[y_out == 0], X_out[y_out == 1]
plt.scatter(X_in_0[:, 0], X_in_0[:, 1], s=s, c='C0', marker='o', label='Inlier, class 0')
plt.scatter(X_in_1[:, 0], X_in_1[:, 1], s=s, c='C1', marker='o', label='Inlier, class 1')
plt.scatter(X_out_0[:, 0], X_out_0[:, 1], s=s, c='C0', marker='v', label='Outlier, class 0')
plt.scatter(X_out_1[:, 0], X_out_1[:, 1], s=s, c='C1', marker='v', label='OUtlier, class 1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of Training Data')
plt.legend(loc='best')
plt.show()
