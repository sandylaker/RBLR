from rblr import *
X_train, y_train, X_test, y_test = simulation_setup(n_i=1000, n_o=200, n_t=1000, p=8,
                                                          sigma_e=0.25)


stbp =StratifiedBootstrap(warm_start=True, fit_intercept=True)
t1 = time.time()
stbp.fit(X_train, y_train, n_bootstrap=10, n_strata=10, sort='residual', verbose=True)
print("Consumed time of Stratified Bootstrap fit %.2f s" % (time.time() - t1))

lr = LogisticRegression(fit_intercept=True, solver='lbfgs')
lr.fit(X_train, y_train)

clbp = ClassicalBootstrap()
clbp.fit(X_train, y_train, n_bootstrap=10)

print("Classical LR accuracy: ", lr.score(X_test, y_test))
print("Classical Bootstrap accuracy: ", clbp.score(X_test, y_test))
print("Stratified Bootstrap accuracy: ", stbp.score(X_test, y_test))