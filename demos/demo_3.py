import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('MacOSX')
from rblr.influence_function_bootstrap import IFB
from rblr.classical_bootstrap import ClassicalBootstrap
from sklearn.linear_model import LogisticRegression
from rblr.simulation_setup import simulation_setup


matplotlib.rcParams['font.size'] = 10

q = np.linspace(0.1, 1, 20)
b = 10
X_train, y_train, X_test, y_test = simulation_setup(n_i=1000, n_o=200, p=8)

lr = LogisticRegression(solver='lbfgs')
lr.fit(X_train, y_train)
score_lr = lr.score(X_test, y_test)

class_boot = ClassicalBootstrap()
class_boot.fit(X_train, y_train, n_bootstrap=b)
score_class_boot = class_boot.score(X_test, y_test)

score_ifb = []
for q_ in q:
    ifb = IFB(c=None, gamma=5)
    ifb.fit(X_train, y_train, n_bootstrap=b, quantile_factor=q_)
    score_ifb.append(ifb.score(X_test, y_test))

f1 = plt.figure(1, figsize=(7, 4.8))
plt.plot(q, score_ifb, label='IFB')
plt.hlines(score_lr, q[0], q[-1], linestyles='dashed', colors='r', label='Classical LR')
plt.hlines(score_class_boot, q[0], q[-1], linestyles='dashed', colors='g', label='Classical '
                                                                                 'Bootstrap')
plt.xlabel('percentile of distribution', )
plt.ylabel('Accuracy',)
plt.title('IFB with different tuning constants')
plt.legend(loc='center left')
plt.show()

