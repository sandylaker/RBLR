from rblr.simulation_setup import simulation_setup
from rblr.mestimator import MEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
plt.style.use('ggplot')

n_i = 10000
n_o = 1000
p = 20

data_train, data_test, beta = simulation_setup(
    n_i=n_i, n_o=n_o, p=p, sigma_o=100, sigma_e=0.1)
df_train = pd.DataFrame(data=data_train)
df_test = pd.DataFrame(data=data_test)
# df_train = pd.read_csv('data_train.csv')

df_train[p] = df_train[p].astype(int)

# store the numpy array representation of the dataframe
X = df_train.drop([p], axis=1).values

me = MEstimator()
# estimated location of inliers
loc_estimated = me.loc_estimator(X)
scale_estimated = me.scale_estimator(X)
# NOTICE here X is centered
X = X - loc_estimated
X_norm = np.linalg.norm(X, axis=1)

# use M-estimator of scale to compute the criteria of selecting inliers
T = 4 * np.sqrt((np.log(p) + np.log(n_i))) * np.max(scale_estimated)
# get the row numbers of inliers
inliner_marker = np.where(X_norm < T)[0]
print('deleted outliers: %d of total train observations %d' %
      (len(X) - len(inliner_marker), len(X)))

# select inliers
df_train_in = df_train.iloc[inliner_marker]

# set X,y training data for LR regressor
clf = LogisticRegression(solver='liblinear')
X_train = df_train_in.drop([p], axis=1)
y_train = df_train_in[p]

# train the regressor
clf.fit(X_train, y_train)

# load and prepare the test data
# df_test = pd.read_csv('data_test.csv')
X_test = df_test.drop([p], axis=1)
y_test = df_test[p].astype(int)

y_predict = clf.predict(X_test)
print("With preprocessing: \n", classification_report(y_test, y_predict,))

# logistic regression without preprocessing
clf_no_prep = LogisticRegression(solver='liblinear')
clf_no_prep.fit(df_train.drop([p], axis=1), df_train[p])
y_predict_no_prep = clf_no_prep.predict(X_test)
print(
    "Without preprocessing: \n",
    classification_report(
        y_test,
        y_predict_no_prep))
