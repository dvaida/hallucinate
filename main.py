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
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

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


def build_all_experiment(name, train_df, test_df, cv_shuffle):
    exp = Experiment(name=name, cv=30, cv_shuffle=cv_shuffle, sc='accuracy', parallel=True)

    bare_feature_names = ['Pclass', 'Sex', 'Fare', 'SibSp', 'Parch', 'Cabin', 'Embarked', 'Title',
                          'Ticket']
    hallucinated_feature_names = ['Fare^2', 'AgeFareRatio', 'FarePerPerson', 'FamilySize', 'Single',
                                  'SmallFamily', 'LargeFamily', 'AgeFareRatio^2', 'FareD', 'FareD2',
                                  'FarePerPerson^2', 'Surname', 'AgeD']

    bare_features = exp.make_features(bare_feature_names, name='Bare Features', train_data=train_df,
                                      test_data=test_df, target='Survived')

    hallucinated_features = exp.make_features(hallucinated_feature_names,
                                              name='Hallucinated', train_data=train_df,
                                              test_data=test_df, parent=bare_features,
                                              target='Survived')

    bare_features.set_feature_selector(
        SelectFromModel(DecisionTreeClassifier(random_state=7, max_depth=12), threshold=0.0005))
    hallucinated_features.set_feature_selector(
        SelectFromModel(DecisionTreeClassifier(random_state=7, max_depth=12), threshold=0.0005))

    bare_features.transform(['Pclass'], 'onehot')
    bare_features.transform(['Sex'], 'map', mapping={'male': 0, 'female': 1})
    bare_features.transform(['Fare'], 'fillna', strategy='mean')

    bare_features.transform(['Cabin'], 'fillna', strategy='value', value='XXX')
    bare_features.transform(['Cabin'], 'method', method_handle=Transforms.extract_cabin)
    bare_features.transform(['Cabin'], 'onehot')

    bare_features.transform(['Embarked'], 'fillna', strategy='value', value='S')
    bare_features.transform(['Embarked'], 'onehot')

    bare_features.transform(['Title'], 'method', from_=['Name'],
                            method_handle=Transforms.extract_title)
    bare_features.transform(['Title'], 'onehot')

    bare_features.transform(['Ticket'], 'method', from_=['Ticket'],
                            method_handle=Transforms.extract_ticket)
    bare_features.transform(['Ticket'], 'onehot')

    hallucinated_features.transform(['Fare'], 'fillna', strategy='mean')
    hallucinated_features.transform(['FareD'], 'discretize', from_=['Fare'],
                                    values_range=[-0.1, 10, 30])
    hallucinated_features.transform(['FareD'], 'onehot')

    hallucinated_features.transform(['FareD2'], 'discretize', from_=['Fare'], q=20)
    hallucinated_features.transform(['FareD2'], 'onehot')

    hallucinated_features.transform(['Fare^2'], '*', from_=['Fare, Fare'])

    # TODO this will look much better with an ExpressionTransform(expr='SibSp + Parch + 1')
    hallucinated_features.transform(['FamilySize'], '+', from_=['SibSp,Parch,1'])
    hallucinated_features.transform(['Single'], 'map', from_=['FamilySize'],
                                    mapping={1: 1, '_others': 0})
    hallucinated_features.transform(['SmallFamily'], 'map', from_=['FamilySize'],
                                    mapping={2: 1, 3: 1, 4: 1, '_others': 0})
    hallucinated_features.transform(['LargeFamily'], 'map', from_=['FamilySize'],
                                    mapping={1: 0, 2: 0, 3: 0, 4: 0, '_others': 1})

    hallucinated_features.transform(['FarePerPerson'], '/', from_=['Fare,FamilySize'])
    hallucinated_features.transform(['FarePerPerson^2'], '*', from_=['FarePerPerson,FarePerPerson'])

    # t1.transform(['Age'], 'method', method_handle=Transforms.predict_age)
    hallucinated_features.transform(['Age'], 'fillna', strategy='median')

    hallucinated_features.transform(['TmpFare'], 'fillna', from_=['Fare'], strategy='mean')
    hallucinated_features.transform(['TmpFare'], '+', from_=['TmpFare,0.1'])
    hallucinated_features.transform(['AgeFareRatio'], '/', from_=['Age,TmpFare'])
    hallucinated_features.transform(['AgeFareRatio^2'], '*', from_=['AgeFareRatio,AgeFareRatio'])

    hallucinated_features.transform(['AgeD'], 'discretize', from_=['Age'],
                                    values_range=[0, 6, 13, 19, 25, 35, 60])
    hallucinated_features.transform(['AgeD'], 'onehot')

    hallucinated_features.transform(['Surname'], 'method', from_=['Name,FamilySize'],
                                    method_handle=Transforms.extract_surname)
    hallucinated_features.transform(['Surname'], 'onehot')

    # t1.transform(['FareC', 'Fare^2', 'AgeFareRatio', 'FarePerPerson'], 'log')
    # t1.transform(['FareC', 'Fare^2', 'AgeFareRatio', 'FarePerPerson'], 'std')

    exp.add_estimator(EstimatorConfig(DecisionTreeClassifier(), {}, 'DTR'))
    # exp.add_estimator(EstimatorConfig(XGBClassifier(nthread=1), {}, 'XGB'))
    # exp.add_estimator(EstimatorConfig(LGBMClassifier(nthread=1), {}, 'LGB'))
    exp.add_estimator(EstimatorConfig(LogisticRegression(n_jobs=1), {}, 'LR'))
    # exp.add_estimator(EstimatorConfig(KNeighborsClassifier(n_jobs=1), {}, 'KNN'))
    exp.add_estimator(EstimatorConfig(GaussianNB(), {}, 'GNB'))
    # exp.add_estimator(EstimatorConfig(VotingBuilder(exp.non_stacking_configs()), {}, 'VOT',
    #                                 stacking=True))

    return exp, bare_features


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
    # exp2.grid_search_all(f_sel_thresholds=dtr_f_sel_thresholds)
    exp2.grid_search_all()
    print('Best run: \n\n{}'.format(exp2.find_best_run('LR')))
    # create_submission(exp2, t2)
    kaggle = Kaggle(exp2, 'PassengerId')
    submissions = kaggle.create_submissions()
    submissions = pd.concat([submissions, pd.read_csv('titanic/test_complete.csv')[['Survived']]],
                            axis=1)
    print('Correlations with real values:\n')
    print(submissions.corr()[submissions.corr()['Survived'] > 0.5]['Survived'].sort_values(
        ascending=False))
    exp2.plot_cv_runs()
    exp2.plot_f_sel_learning_curve()
    exp2.plot_feature_importance()
    exp2.plot_correlations()
    plt.show()
