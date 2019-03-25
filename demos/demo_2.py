from sklearn.linear_model import LogisticRegression
import numpy as np
from rblr.simulation_setup import simulation_setup
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("MacOSX")

font = {
    'family': 'Arial',
    'size': 16
}
np.random.seed(0)
X_in_0 = np.random.multivariate_normal(np.zeros(2), cov=np.eye(2), size=100)
y_in_0 = np.zeros(100, dtype=int)
X_in_1 = np.random.multivariate_normal(np.array([2, 2]),cov=np.eye(2), size=100)
y_in_1 = np.ones(100, dtype=int)

# outliers
X_out = np.random.multivariate_normal(np.array([200, 5]), cov=np.eye(2), size=200)
y_out = np.zeros(200, dtype=int)

# X_train = np.concatenate((X_in_0, X_in_1), axis=0)
# y_train = np.concatenate((y_in_0, y_in_1))
X_train = np.concatenate((X_in_0, X_in_1, X_out), axis=0)
y_train = np.concatenate((y_in_0, y_in_1, y_out))



clf = LogisticRegression(solver='lbfgs',)
clf.fit(X_train, y_train)
beta_fit = clf.coef_[0]

x_ = np.linspace(-5, 5, 100)
y_ = - 1 / beta_fit[1] *(beta_fit[0] * x_ + clf.intercept_[0])

f1 = plt.figure(1)
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], c='r', label='Class 0',
            )
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], c='b', label='Class 1',
            )
plt.plot(x_, y_, label='Fitted Boundary', color='g')
plt.xlabel('Feature 1', fontdict=font)
plt.ylabel('Feature 2', fontdict=font)
plt.title('LR on contaminated data', fontdict=font)
plt.xlim(-5, 5)
plt.legend()
plt.show()
