logreg_params = {
        'C': [0.001, 0.01, 0.1, 1, 5, 10, 50, 100],
        # 'solver': ['liblinear'],
        # 'penalty': ['l1', 'l2'],
        # 'tol': [0.0001, 0.001, 0.01, 0.1, 1]
}

rforest_params = {
    'n_estimators': [10, 20, 50],
    'max_features': ['log2', 'sqrt', 'auto'],
    'criterion': ['entropy', 'gini'],
    'max_depth': [6, 10, 12],
    'min_samples_split': [3],
    'min_samples_leaf': [1]
}

# xgb_params = {
#     'max_depth': [3, 6],
#     'learning_rate': [0.3,],
#     'n_estimators': [50, 75, 85],
#     'gamma': [0.1, 0.2, 0.4],
#     'reg_alpha': [0.5, 0.6, 0.9],
#     'reg_lambda': [0.5, 1],
#     'colsample_bytree': [0.9],
#     'colsample_bylevel': [1],
#     'subsample': [0.5, 1],
#     'min_child_weight': [3]
# }

xgb_params = {
    'max_depth': [3, 6],
    'learning_rate': [0.3,],
    'n_estimators': [50,],
    'gamma': [0.4],
    'reg_alpha': [0.6,],
    'reg_lambda': [0.1, 0.5, 1],
    'colsample_bytree': [0.9],
    'colsample_bylevel': [1],
    'subsample': [0.9, 1],
    'min_child_weight': [3]
}

svc_params = {
    'kernel': ['rbf'],
    'gamma': [0.001, 0.01, 0.1, 1],
    'C': [1, 10, 100, 1000],
    'tol': [0.001, 0.01, 0.1, 1, 10]
}

knn_params = {
    'n_neighbors': [3, 5, 10, 15, 20]
}

adb_params = {
    'algorithm': ['SAMME.R'],
    'n_estimators': [5, 6, 10, 20],
    'learning_rate': [.1, .5, 1., 3, 5]
}

xtrees_params = {
    'n_estimators': [10, 50, 75],
    'max_features': ['log2', 'sqrt', 'auto'],
    'criterion': ['entropy', 'gini'],
    'max_depth': [3, 6, 9, 10, 12],
    'min_samples_split': [3],
    'min_samples_leaf': [1]
}

lgbm_params = {
    'learning_rate': [0.01, 0.1, 1, 10, 100],
    'n_estimators': [10, 20, 40, 50, 100]
    # 'learning_rate': [0.1],
    # 'n_estimators': [100],
}

mlp_params = {
    'hidden_layer_sizes': [(10,), (20,), (30,)],
    # 'activation': ['logistic', 'tanh', 'relu'],
    'activation': ['tanh'],
    # 'solver': ['lbfgs', 'sgd', 'adam']
    'solver': ['lbfgs']
}

per_params = {
    'penalty': ['l2', 'l1', 'elasticnet'],
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10],
    'n_iter': [10, 50, 100, 200, 500, 1000],
    'eta0': [0.1, 1, 10]
}

sgd_params = {
    'penalty': ['l2', 'l1', 'elasticnet'],
    'alpha': [0.0001],  # , 0.001, 0.01, 0.1, 1, 10],
    'n_iter': [500],  # 10, 50, 100, 200, 1000],
    'eta0': [1],
    'loss': ['hinge', 'log'],
    'epsilon': [0.1]  # 0.001, 1, 10]
}

dtr_params = {
    'criterion':['gini', 'entropy'],
    'max_depth': [3, 6, 9]
}
