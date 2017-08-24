### A lightweight library for quick, iterative ML experimentation ###

Head over [here](https://github.com/dvaida/hallucinate/blob/master/Teaser.ipynb) to see the code below as a notebook

### Teaser code: ###

```python
train_df = pd.read_csv('titanic/train.csv')
test_df = pd.read_csv('titanic/test.csv')
feature_names = train_df.drop(['PassengerId', 'Survived', 'Name', 'Ticket'], 1).columns.values.tolist()
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
