import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from rblr.simulation_setup import simulation_setup
from rblr.util_funcs import sigmoid

class ClassicalBootstrap:

    def __init__(self, fit_intercept=False, solver='lbfgs', **kwargs):
        self.clf_lr = LogisticRegression(fit_intercept=fit_intercept, solver=solver)
        self.beta = None

    def resample(self, X, n_bootstrap, n_samples_each_bootstrap):
        sample_index_range = X.shape[0]
        sample_index = np.random.choice(sample_index_range,
                                        size=(n_bootstrap, n_samples_each_bootstrap))
        return X[sample_index]

    def fit(self, X, y, n_bootstrap=20, n_samples_each_bootstrap=None):
        y = np.asarray(y).astype(int)
        if n_samples_each_bootstrap is None:
            n_samples_each_bootstrap = X.shape[0]
        if y.ndim == 1:
            y = np.array(y)[:, np.newaxis]

        # concatenate X and y for convinience
        X_concat = np.concatenate((X, y), axis=1)
        bootstrap_samples = self.resample(X_concat, n_bootstrap, n_samples_each_bootstrap)
        beta = []
        for X_concat_b in bootstrap_samples:
            X_b = X_concat_b[:, :-1]
            y_b = X_concat_b[:, -1]
            self.clf_lr.fit(X_b, y_b)
            beta.append(self.clf_lr.coef_[0])
        self.beta = np.mean(beta, axis=0)
        return self

    def predict(self, X, prob_threshold=0.5):
        """
        predict labels for test data

        :param X: ndarray, shape(n_samples, n_features)
        data to be predicted

        :param prob_threshold: double, default: 0.5,
                            probability threshold for determining the predicted labels.

        :return: ndarray, shape(n_samples,)
                predicted labels, which are coded with {1,0}
        """
        ss = StandardScaler()
        X = ss.fit_transform(X)
        if self.beta is None:
            raise ValueError("Model is not fitted yet")
        predict_prob = sigmoid(np.dot(X, self.beta))
        return np.array(predict_prob >= prob_threshold, dtype=int)

    def score(self, X_test, y_test, prob_threshold=0.5):
        """
        computes accuracy of the model

        :param X_test: ndarray, shape(n_samples, n_features)
                    Test data

        :param y_test: ndarray, shape(n_samples,)
                    Labels of test data

        :param prob_threshold: double, default: 0.5
                    probability threshold for determining the predicted labels

        :return: double
                accuracy of the logistic regression model.
        """
        y_test = np.array(y_test, dtype=int)
        y_predict = self.predict(X_test, prob_threshold=prob_threshold)
        accuracy = np.mean(y_predict == y_test)
        return accuracy



if __name__ == '__main__':
    data_train, data_test, beta_actual = simulation_setup(n_i=1000, n_o=200, n_t=1000, p=10,
                                                          sigma_e=0.25)
    X_train, y_train = data_train[:, :-1], data_train[:, -1]
    X_test, y_test = data_test[:, :-1], data_test[:, -1]
    classical_bootstrap = ClassicalBootstrap()
    classical_bootstrap.fit(X_train, y_train)
    print("classical bootstrap score: ", classical_bootstrap.score(X_test, y_test))

