import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('MacOSX')
from rblr.influence_function_bootstrap import IFB
from rblr.classical_bootstrap import ClassicalBootstrap
from sklearn.linear_model import LogisticRegression
from rblr.preprocessing import Preprocessor
from rblr.simulation_setup import simulation_setup
from rblr import MEstimator

matplotlib.rcParams['font.size'] = 10

X = np.concatenate((np.random.multivariate_normal(np.zeros(2),  2 * np.eye(2), 50),
                    np.random.multivariate_normal(10 * np.ones(2), 2 * np.eye(2), 10)),
                   axis=0)
mestimator = MEstimator()

n_total = X.shape[0]
p = X.shape[1]

X_centered = X - mestimator.loc_estimator(X)
scale_estimated = mestimator.scale_estimator(X)
X_norm = np.linalg.norm(X_centered, axis=1)

# NOTICE: n_inlier should be used, however, the number of inliers is
# NOTICE: usually unkown, so we use 0.5*n_total to estimate the maximal number of inliers.
n_inliers = 0.5 * n_total

T = 4 * np.sqrt(np.log(p) + np.log(n_inliers)) * np.max(scale_estimated)
inlier_flag = np.less_equal(X_norm, T)
X_in = X[inlier_flag]
X_out = X[~inlier_flag]

theta = np.linspace(0, 2*np.pi, 1000)
circle = np.array([np.cos(theta), np.sin(theta)]) * T

f1 = plt.figure(1)
plt.scatter(X_in[:, 0], X_in[:, 1], c='C0', label='Inliers')
plt.scatter(X_out[:, 0], X_out[:, 1], c='C1', label='Outliers')
plt.plot(circle[0], circle[1], c='C3', label='T')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Preprocessing Procedure')
plt.legend()
plt.show()