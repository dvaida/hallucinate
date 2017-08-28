### A lightweight library for quick, iterative ML experimentation ###

Head over [here](https://github.com/dvaida/hallucinate/blob/master/Teaser.ipynb) to see the code below as a notebook

### Teaser code: ###

```python
train_df = pd.read_csv('titanic/train.csv')
test_df = pd.read_csv('titanic/test.csv')
feature_names = train_df.drop(['PassengerId', 'Survived', 'Name', 'Ticket'], 1).columns.values
experiment = hl.Experiment(name='Titanic', cv=30, sc='accuracy', cv_shuffle=False, parallel=True)
features = experiment.make_features(name='Default Features',
                                    features=feature_names,
                                    train_data=train_df, 
                                    test_data=test_df,
                                    target='Survived')
def extract_cabin(df):
    return df['Cabin'].apply(lambda x: x[0])

# Categorical
features.transform(['Sex'], 'map', mapping={'male': 0, 'female': 1})
features.transform(['Cabin'], 'fillna', strategy='value', value='XXX')
features.transform(['Cabin'], 'method', method_handle=extract_cabin)
features.transform(['Embarked'], 'fillna', strategy='value', value='S') # most frequent value

# Numerical
features.transform(['Age'], 'fillna', strategy='median')
features.transform(['Fare'], 'fillna', strategy='mean')

# All
features.transform(['Pclass', 'Cabin', 'Embarked'], 'onehot')

experiment.grid_search_all(verbose=True)
experiment.plot_cv_runs()
```

### Complete Usage (until I get the docs done)

```
import pandas as pd
from hallucinate import Experiment, EstimatorConfig, Kaggle
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier

train_df = pd.read_csv('titanic/train.csv')
test_df = pd.read_csv('titanic/test.csv')

# We'll need this in a bit
def extract_cabin(df):
    return df['Cabin'].apply(lambda x: x[0])

# Make an experiment
experiment = Experiment(name='Titanic', cv=30, sc='accuracy', cv_shuffle=False, parallel=True)

# Make a feature set and some feature engineering (you can define one or more of them, )
feature_names = train_df.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Parch'], 1).columns.values
features = experiment.make_features(feature_names, train_df, test_df, name='Default', target='Survived')
features.transform(['Sex'], 'map', mapping={'male': 0, 'female': 1})
features.transform(['Cabin'], 'fillna', strategy='value', value='XXX')
features.transform(['Cabin'], 'method', method_handle=extract_cabin)
features.transform(['Embarked'], 'fillna', strategy='value', value='S') # most frequent value
features.transform(['Age'], 'fillna', strategy='median')
features.transform(['Fare'], 'fillna', strategy='mean')
features.transform(['Pclass', 'Cabin', 'Embarked'], 'onehot', drop_first=True)

# Add some classfier configurations
grid_search_params = {'max_depth': [3, 6, 9], 'n_estimators': [10, 20, 50]}
experiment.add_estimator(EstimatorConfig(RandomForestClassifier(), grid_search_params, 'RFE'))
experiment.add_estimator(EstimatorConfig(ExtraTreesClassifier(), grid_search_params, 'XTR'))
experiment.add_estimator(EstimatorConfig(LGBMClassifier(), grid_search_params, 'LGB'))

# Run a grid search on all the classifiers (in parallel if parallel=True for the Experiment)
experiment.grid_search_all()

# See how that looks, overall
experiment.overview()

# See the correlations, maybe there are some redundant / highly correlated features that you want to rig
experiment.plot_correlations(figsize=11)

# Let's see which features matter to whom
experiment.plot_feature_importance()

# You want to include the Name - w/o any feature engineering ATM, just to make another point
# a bit later - automatic feature selection
# At the same time you don't want to write/duplicate the previous feature engineering code all over again.
# So, you set the previous feature set as a parent
features_w_name = experiment.make_features(['Name'], train_df, test_df, name='W/ Name', target='Survived', parent=features)
features_w_name.transform(['Name'], 'onehot')
features_w_name

# Now there are two feature sets in the experiment and you can see them both
experiment.overview()

# Run grid search on all classifiers, and all feature sets
experiment.grid_search_all()

# Plot the CV accuracy for every classfier, compare between feature sets
experiment.plot_cv_runs()

# Oops, from the previous run you see that one-hot encoded Name just bumped the feature used to 1323 and destroyed performance.
# Too many of them, curse of dimensionality etc. let's rig some out
features_w_name.set_feature_selector(SelectFromModel(DecisionTreeClassifier()))
experiment.grid_search_all()

# Hmm, it gets better with feature selection, let's see if different feature selection thresholds make a difference
experiment.grid_search_all(f_sel_thresholds=[0.0005, 0.001])

# Visualize a learning curve for the different feature selection thresholds
experiment.plot_f_sel_learning_curve()

# Finally, create Kaggle submissions from the experiment
s = Kaggle(experiment, index_feature='PassengerId').create_submissions()
