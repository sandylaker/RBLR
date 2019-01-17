import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

n, n_1 = int(1e5), int(5e4)
p = 20
multi_mean = np.zeros(p)
multi_cov = np.eye(p)

X_good = np.random.multivariate_normal(multi_mean, multi_cov, size=n)*100 + 100000
# outlier = np.random.multivariate_normal(multi_mean, multi_cov, size=n_1)*1 + 10
outlier = np.random.uniform(-1, 1, (n_1, p))*100 + 1e8
if len(outlier) != 0:
    X = np.concatenate((X_good, outlier), axis=0)
else:
    X = X_good

# scaler = StandardScaler()
# X = scaler.fit_transform(X)
X = X - np.median(X, axis=0)

print("Probability: %.2f" % (1 - p**(-2)))
X_norm = np.linalg.norm(X, axis=1)
print("max(X_norm): %f" % np.max(X_norm))

T = 4 * np.sqrt((np.log(p) + np.log(n)))
T = T * np.max(np.median(np.abs(X), axis=0))

X_in = X[np.where(X_norm < T)[0]]
X_out = X[np.where(X_norm >= T)[0]]
print("inlier: %d ; outlier: %d" % (len(X_in), len(X_out)))

theta = np.linspace(0, 2*np.pi, 1000)
circle = np.array([np.cos(theta), np.sin(theta)]) * T

if p == 2:
    alpha = 0.5
    plt.scatter(X_in[:, 0], X_in[:, 1], s=0.5, alpha=alpha, c='blue')
    plt.scatter(X_out[:, 0], X_out[:, 1], s=0.5, alpha=alpha, c='red')
    plt.plot(circle[0], circle[1], linewidth=0.8, color='green')
    plt.show()
