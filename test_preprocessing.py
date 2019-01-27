from mestimator import MEstimator
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
plt.style.use('ggplot')


n = 10000
p = 20

df_train = pd.read_csv('data_train.csv')
df_train['20'] = df_train['20'].astype(int)

# store the numpy array representation of the dataframe
X = df_train.drop(['20'], axis=1).values

me = MEstimator()
# estimated location of inliers
loc_estimated = me.loc_estimator(X)
scale_estimated = me.scale_estimator(X)
X = X - loc_estimated
X_norm = np.linalg.norm(X, axis=1)

# use M-estimator of scale to compute the criteria of selecting inliers
T = 4 * np.sqrt((np.log(p) + np.log(n))) * np.max(scale_estimated)
inliner_marker = np.where(X_norm < T)[0]

# select inliers
df_train_in = df_train.iloc[inliner_marker]

# set X,y training data for LR regressor
clf = LogisticRegression(solver='liblinear')
X_train = df_train_in.drop(['20'], axis=1)
y_train = df_train_in['20']

# train the regressor
clf.fit(X_train, y_train)

# load and prepare the test data
df_test = pd.read_csv('data_test.csv')
X_test = df_test.drop(['20'], axis=1)
y_test = df_test['20'].astype(int)

score = clf.score(X_test, y_test)
print('score: %f' % score)

y_predict = clf.predict(X_test)
print("With preprocessing: \n", classification_report(y_test, y_predict,))

# logistic regression without preprocessing
clf_no_prep = LogisticRegression(solver='liblinear')
clf_no_prep.fit(df_train.drop(['20'], axis=1), df_train['20'])
y_predict_no_prep = clf_no_prep.predict(X_test)
print("Without preprocessing: \n", classification_report(y_test, y_predict_no_prep))


