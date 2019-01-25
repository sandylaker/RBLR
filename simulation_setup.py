import numpy as np
import pandas as pd
from util_funcs import sigmoid

# hyper parameters
# number good samples, outliers, test samples
n_i, n_o, n_t = 1000, 100, 1000
# dimension of data
p = 20
# standard deviation of gaussian noise
sigma_e = 0.5
# range of outliers
sigma_o = 10

np.random.seed(100)


# sample beta, here size=1 will generate a 2-D array of shape(1,p),
# the flat array will be used, so the first component is extracted
beta = np.random.multivariate_normal(np.zeros(p), np.eye(p), size=1)
# normalize beta
beta = beta[0] / np.linalg.norm(beta)

# sample X_i of good data
X_i = np.random.multivariate_normal(np.zeros(p), np.eye(p), size=n_i)
# gaussian noise of good samples
noise_i = np.random.randn(n_i)

# generate labels of good samples
tau = sigmoid(np.dot(beta, X_i.T) + noise_i)
y_i = np.array(np.less_equal(tau, np.random.uniform(0, 1, n_i)), dtype=np.int)

# test sample
X_t = np.random.multivariate_normal(np.zeros(p), np.eye(p), size=n_t)
# generate label of test samples
y_t = np.greater_equal(np.dot(beta, X_t.T) + np.random.randn(n_t), 0).astype(int)

# sample X_o of outliers
X_o = np.random.uniform(- sigma_o, sigma_o, (n_o, p))
# gaussian noise of outliers
noise_o = np.random.randn(n_o)
# generate labels of outliers
y_o = np.greater_equal(np.dot(- beta, X_o.T) + noise_o, 0).astype(int)

# concatenate the data into a dataframe
y_i = y_i[:, np.newaxis]
df_i = pd.DataFrame(data=np.concatenate((X_i, y_i), axis=1))

y_t = y_t[:, np.newaxis]
df_t = pd.DataFrame(data=np.concatenate((X_t, y_t), axis=1))

y_o = y_o[:, np.newaxis]
df_o = pd.DataFrame(data=np.concatenate((X_o, y_o), axis=1))

# concatenate the good data and oulier data into a dataframe for training
df_train = pd.concat([df_i, df_o])

# write into csv
df_train.to_csv('data_train.csv', index=False)
df_t.to_csv('data_test.csv', index= False)