try:
    import matplotlib
    import sys

    matplotlib.use('TkAgg' if 'darwin' in sys.platform else 'Agg')
    import matplotlib.pyplot as plt
except Exception as ex:
    print(ex)
    print("Matplotlib unavailable")

import os
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier

from hallucinate import *


class Transforms(object):
    @staticmethod
    def extract_cabin(df):
        return df['Cabin'].apply(lambda x: x[0])

    @staticmethod
    def extract_ticket(df):
        def ret_ticket(ticket):
            ticket = ticket.replace('.', '')
            ticket = ticket.replace('/', '')
            ticket = ticket.split()
            ticket = map(lambda t: t.strip(), ticket)
            ticket = [a for a in filter(lambda t: not t.isdigit(), ticket)]
            if len(ticket) > 0:
                return ticket[0]
            else:
                return 'XXX'

        return df['Ticket'].apply(ret_ticket)

    @staticmethod
    def extract_title(df):
        return df['Name'].apply(lambda x: x.split('.')[0]).apply(lambda x: x.split(' ')[-1])

    @staticmethod
    def extract_surname(df):
        def do_it(a):
            return a[0].split(',')[0] if a[1] > 1 else 'UNK'

        return df[['Name', 'FamilySize']].apply(do_it, axis=1)

    @staticmethod
    def predict_age(df):
        feats = ['Pclass', 'Sex', 'FareC', 'Ticket', 'Title']
        fs = Features(feats,
                      target='Age',
                      train_data=pd.DataFrame(df[~df['Age'].isnull()].copy()),
                      test_data=pd.DataFrame(df[df['Age'].isnull()].copy()))
        fs.transform(['Pclass', 'Ticket', 'Title'], 'onehot')
        X_train = fs.preprocess().values
        y_train = df[~df['Age'].isnull()]['Age'].values
        regressor = SVR(kernel='rbf')
        regressor.fit(X_train, y_train)
        X_pred = fs.preprocess(for_test=True).values
        y_pred = regressor.predict(X_pred)
        df.ix[df['Age'].isnull(), 'Age'] = y_pred.astype(int)
        return df['Age']


def create_submission(experiment, train_fs):
    print(' -- Creating submissions...\n')
    os.makedirs('submissions', exist_ok=True)

    all_predictions = {}

    for config in experiment.configs:
        cv_runs = []
        for run in experiment.runs[config.name]:
            cv_runs.append(run.prediction_results)
            model = run.best_model
            fs_name = run.key()
            train_fs.feature_selector = None
            X, y, features, sel_features = train_fs.build_Xy()
            model.fit(X, y)
            # We need to copy after fitting, in case we have a feature selector
            X_test, features, sel_features = train_fs.build_X(for_test=True)
            predictions = model.predict(X_test)
            all_predictions['{}_{}'.format(config.name, fs_name)] = predictions
            pd.concat([train_fs.get_test_data()['PassengerId'],
                       pd.Series(predictions.astype(int), name='Survived')], axis=1).to_csv(
                'submissions/submission_{}_{}.csv'.format(config.name, fs_name), index=False)
        stk = np.asarray(cv_runs)
        stk = stk.reshape((stk.shape[0] * stk.shape[1], stk.shape[2]))
        stk = np.mean(stk, axis=0)
        all_predictions['{}_STK'.format(config.name)] = [int(z) for z in stk]
        pd.concat([train_fs.get_test_data()['PassengerId'],
                   pd.Series(stk.astype(int), name='Survived')], axis=1).to_csv(
            'submissions/submission_{}_{}.csv'.format(config.name, 'STK'), index=False)
        # print('{}: {}'.format(config.name, stk.shape))
    print(' -- Done creating submissions.\n')

    if len(all_predictions) > 1:
        print('\nPrediction correlations:\n')
        all_predictions['REAL'] = pd.read_csv('titanic/test_complete.csv')['Survived'].values
        tmp = pd.DataFrame(all_predictions)
        print(tmp.corr()[tmp.corr()['REAL'] > 0.52]['REAL'].sort_values(ascending=False))
        print('-- Wrote predictions including the real values in all_predictions.csv')


def build_all_experiment(name, train_df, test_df, cv_shuffle):
    exp = Experiment(name=name, cv=30, cv_shuffle=cv_shuffle, sc='accuracy', parallel=True)

    features = ['Pclass', 'Sex', 'FareC', 'FareD', 'Fare^2', 'AgeFareRatio', 'FarePerPerson',
                'AgeD', 'FamilySize', 'SibSp', 'Parch', 'Single', 'SmallFamily', 'LargeFamily',
                'Cabin', 'Embarked', 'Title', 'Surname', 'Ticket', 'AgeFareRatio^2', 'FareD2']

    t1 = exp.make_features(features, name='All', train_data=train_df, test_data=test_df,
                           target='Survived')

    # t1.set_dim_reducer(KernelPCA(n_components=60, kernel='rbf'))
    # t1.set_dim_reducer(SelectFromModel(XGBClassifier(nthread=1), threshold=0.001))
    # t1.set_dim_reducer(SelectFromModel(LGBMClassifier(nthread=1), threshold=0.005))
    # t1.set_dim_reducer(SelectFromModel(LogisticRegression(n_jobs=1), threshold=0.15))
    t1.set_feature_selector(
        SelectFromModel(DecisionTreeClassifier(random_state=7, max_depth=12), threshold=0.0005))

    t1.transform(['Fare'], 'fillna', strategy='mean')
    # t1.transform(['Age'], 'fillna', strategy='median')

    t1.transform(['Pclass'], 'onehot')
    t1.transform(['Sex'], 'map', mapping={'male': 0, 'female': 1})
    t1.transform(['FareC'], 'method', from_=['Fare'], method_handle=None)

    t1.transform(['FareD'], 'discretize', from_=['Fare'], values_range=[-0.1, 10, 30])
    t1.transform(['FareD'], 'onehot')

    t1.transform(['FareD2'], 'discretize', from_=['Fare'], q=20)
    t1.transform(['FareD2'], 'onehot')

    t1.transform(['Fare^2'], '*', from_=['Fare, Fare'])

    # TODO this will look much better with an ExpressionTransform(expr='SibSp + Parch + 1')
    t1.transform(['FamilySize'], '+', from_=['SibSp,Parch,1'])
    t1.transform(['Single'], 'map', from_=['FamilySize'], mapping={1: 1, '_others': 0})
    t1.transform(['SmallFamily'], 'map', from_=['FamilySize'],
                 mapping={2: 1, 3: 1, 4: 1, '_others': 0})
    t1.transform(['LargeFamily'], 'map', from_=['FamilySize'],
                 mapping={1: 0, 2: 0, 3: 0, 4: 0, '_others': 1})

    t1.transform(['Cabin'], 'fillna', strategy='value', value='_')
    t1.transform(['Cabin'], 'method', method_handle=Transforms.extract_cabin)
    t1.transform(['Cabin'], 'onehot')

    t1.transform(['Title'], 'method', from_=['Name'], method_handle=Transforms.extract_title)
    t1.transform(['Title'], 'onehot')

    t1.transform(['FarePerPerson'], '/', from_=['Fare,FamilySize'])
    t1.transform(['FarePerPerson^2'], '*', from_=['FarePerPerson,FarePerPerson'])

    t1.transform(['Surname'], 'method', from_=['Name,FamilySize'],
                 method_handle=Transforms.extract_surname)
    t1.transform(['Surname'], 'onehot')

    t1.transform(['Ticket'], 'method', from_=['Ticket'],
                 method_handle=Transforms.extract_ticket)
    t1.transform(['Ticket'], 'onehot')

    t1.transform(['Embarked'], 'fillna', strategy='value', value='C')
    t1.transform(['Embarked'], 'onehot')

    # t1.transform(['Age'], 'method', method_handle=Transforms.predict_age)
    t1.transform(['Age'], 'fillna', strategy='median')

    t1.transform(['TmpFare'], 'fillna', from_=['Fare'], strategy='mean')
    t1.transform(['TmpFare'], '+', from_=['TmpFare,0.1'])
    t1.transform(['AgeFareRatio'], '/', from_=['Age,TmpFare'])
    t1.transform(['AgeFareRatio^2'], '*', from_=['AgeFareRatio,AgeFareRatio'])

    t1.transform(['AgeD'], 'discretize', from_=['Age'], values_range=[0, 6, 13, 19, 25, 35, 60])
    t1.transform(['AgeD'], 'onehot')

    # t1.transform(['FareC', 'Fare^2', 'AgeFareRatio', 'FarePerPerson'], 'log')
    # t1.transform(['FareC', 'Fare^2', 'AgeFareRatio', 'FarePerPerson'], 'std')

    exp.add_estimator(EstimatorConfig(DecisionTreeClassifier(), {}, 'DTR'))
    # exp.add_estimator(EstimatorConfig(XGBClassifier(nthread=1), {}, 'XGB'))
    # exp.add_estimator(EstimatorConfig(LGBMClassifier(nthread=1), {}, 'LGB'))
    # exp.add_estimator(EstimatorConfig(LogisticRegression(n_jobs=1), {}, 'LR'))
    # exp.add_estimator(EstimatorConfig(KNeighborsClassifier(n_jobs=1), {}, 'KNN'))
    # exp.add_estimator(EstimatorConfig(VotingBuilder(exp.non_stacking_configs()), {}, 'VOT',
    #                                 stacking=True))

    return exp, t1


if __name__ == '__main__':
    print()
    train_df = pd.read_csv('titanic/train.csv')
    train_df['Survived'] = train_df['Survived'].astype(int)
    test_df = pd.read_csv('titanic/test.csv')
    test_complete_df = pd.read_csv('titanic/test_complete.csv')
    test_complete_df['Survived'] = test_complete_df['Survived'].astype(int)

    cv_shuffle = False
    # dtr_f_sel_thresholds = [0.0005 + a * 0.0005 for a in range(15)]
    dtr_f_sel_thresholds = [0.0005 + a * 0.0005 for a in range(5)]
    # TODO logistic regression will hang in parallel mode if going <= 0.15 with the threshold
    # Only thing I can correlate with is the training set size. If big, it fucks up silently / hangs
    lr_f_sel_thresholds = [0.1 + a * 0.05 for a in range(5)]
    xgb_f_sel_thresholds = [0.0005 + a * 0.0005 for a in range(8)]
    lgb_f_sel_thresholds = [0.005 + a * 0.005 for a in range(5)]
    print(['{:g}'.format(a) for a in dtr_f_sel_thresholds])
    # exp1, t1 = build_all_experiment('Complete', pd.concat([train_df, test_complete_df]), None,
    #                                 cv_shuffle=cv_shuffle)
    # exp1.show_all_null_stats()
    # plt.show()
    # exp1.grid_search_all(f_sel_thresholds=dtr_f_sel_thresholds)
    # # exp1.plot_results2()
    # exp1.plot_f_sel_learning_curve()

    exp2, t2 = build_all_experiment('Kaggle', train_df, test_df, cv_shuffle=cv_shuffle)
    exp2.overview(verbose=0)
    # print(exp2.features_sources[0].preprocess(include_one_hot=False).head())
    # exp2.show_null_stats(preprocess=True)
    exp2.grid_search_all(f_sel_thresholds=dtr_f_sel_thresholds)
    print('Best run: \n\n{}'.format(exp2.find_best_run('DTR')))
    create_submission(exp2, t2)
    exp2.plot_cv_runs()
    # exp2.plot_f_sel_learning_curve()
    exp2.plot_feature_importance()
    exp2.plot_correlations()
    plt.show()
