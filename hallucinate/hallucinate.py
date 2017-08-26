try:
    import matplotlib
    import matplotlib.pyplot as plt
except Exception as ex:
    print(ex)
    print("Matplotlib unavailable...")

try:
    import seaborn as sns

    sns.set()
    sns.set_style("whitegrid")
except Exception as ex:
    print(ex)
    print("Seaborn unavailable...")

import multiprocessing
import pandas as pd
import numpy as np

from itertools import combinations
from collections import defaultdict
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import get_scorer
from joblib import Parallel, delayed
from mlxtend.classifier import StackingClassifier

from .transform import T_MAP


class EstimatorConfig(object):
    def __init__(self, instance, grid_params, name, cons_params={}, stacking=False):
        self.__instance = instance
        self.grid_params = grid_params
        self.name = name if name else instance.__class__.__name__
        self.stacking = stacking
        self.best_params = {}
        self.cons_params = cons_params

    def build_model(self):
        if "Voting" in self.__instance.__class__.__name__ or "Stacking" in self.__instance.__class__.__name__:
            return self.__instance
        else:
            klass = self.__instance.__class__
            return klass(**({**self.best_params, **self.cons_params}))


class Result(object):
    def __init__(self, instance, best_params, results, prediction_results, feature_source_name,
                 f_sel_threshold):
        self.best_model = instance
        self.best_params = best_params
        self.results = results
        self.mean_score = np.mean(results)
        self.prediction_results = prediction_results
        self.feature_source_name = feature_source_name
        self.f_sel_threshold = f_sel_threshold

    def __repr__(self):
        s = 'Features: {}\nFeature sel threshold: {}\nAvg score: {:.4f}\nCV Results: {}' \
            '\nModel: {}\nBest params: {}'.format(self.feature_source_name, self.f_sel_threshold,
                                                  self.mean_score,
                                                  ['{:.4f}'.format(a) for a in self.results],
                                                  self.best_model, self.best_params)
        return s

    def key(self):
        if self.f_sel_threshold is None:
            return self.feature_source_name
        else:
            return '{}_{:g}'.format(self.feature_source_name, self.f_sel_threshold)


class Features(object):
    def __init__(self, features, train_data=None, test_data=None, target=None, name=None,
                 parent=None):
        self.name = name
        self.features = features
        self.transformations = []
        self.parent = parent
        self.feature_selectors = []
        self.fit_reducers = {}

        # TODO maybe take it from the parent?
        assert train_data is not None

        self._train_data = train_data
        self._test_data = test_data

        # The intent with this is to use both training and test data for feature pre-processing
        # while making the training/validation only on the train_data.
        if test_data is not None:
            self._train_data = pd.concat([train_data, test_data])
        self.training_test_threshold = train_data.shape[0]

        self.target = target if target else parent.target if parent else None
        if not target:
            raise Exception('The first features source in the experiment must specify a target')

        if not self.name:
            self.name = '+'.join([a for a in self._features_including_parents()])

    def set_feature_selector(self, reducer):
        self.feature_selectors.clear()
        self.feature_selectors.append(reducer)

    def add_transformation(self, t):
        self.transformations.append(t)

    def add_all_transformations(self, ts):
        for t in ts:
            self.add_transformation(t)

    def transform(self, feature_names, name, *args, **kwargs):
        tr = T_MAP[name](feature_names, *args, **kwargs)
        self.add_transformation(tr)
        return tr

    def set_selector_threshold(self, new_value):
        if len(self.feature_selectors) == 0:
            return
        if len(self.feature_selectors) != 1:
            raise Exception('Unimplemented yet! You must have a single reducer')
        reducer = self.feature_selectors[0]
        if new_value != 'default':
            reducer.threshold = new_value
        # print('Reducer threshold set to: {}'.format(new_value))
        return reducer.threshold

    def build_Xy(self, warnings=True):
        if self.target is None:
            raise Exception(
                'Specify a target column in the training data set')
        y = self._train_data[self.target].values[: self.training_test_threshold]
        X, feature_names, selected_feature_names = self.build_X(warnings=warnings)

        for reducer in self.feature_selectors:
            if not self.fit_reducers.get(reducer):
                X = reducer.fit_transform(X, y)
                selected_feature_names = feature_names[reducer.get_support()]
                self.fit_reducers[reducer] = True

        return X, y, feature_names, selected_feature_names

    def build_X(self, for_test=False, warnings=True):
        work_df = self.preprocess(for_test)
        if warnings:
            for feature in work_df.columns:
                if work_df[feature].isnull().sum() > 0:
                    print(' !!! WARNING !!! {} has null values'.format(feature))
        X = work_df.values
        feature_names = work_df.columns.values
        selected_feature_names = feature_names
        for reducer in self.feature_selectors:
            if self.fit_reducers.get(reducer):
                X = reducer.transform(X)
                selected_feature_names = feature_names[reducer.get_support()]

        return X, feature_names, selected_feature_names

    def preprocess(self, for_test=False, include_one_hot=True):
        parent_df = pd.DataFrame() if not self.parent else self.parent.preprocess(for_test,
                                                                                  include_one_hot)
        non_one_hot_transforms = [t for t in self.transformations if
                                  'OneHot' not in t.__class__.__name__]
        one_hot_transforms = [t for t in self.transformations if
                              'OneHot' in t.__class__.__name__]
        work_df = pd.DataFrame(self._train_data.copy())
        for t in non_one_hot_transforms:
            work_df = t.apply_to(work_df)

        work_df = work_df[self.features]
        # print(work_df.head())

        if include_one_hot:
            for t in one_hot_transforms:
                work_df = t.apply_to(work_df)

        # parent already has the desired shape
        if for_test:
            work_df = work_df.iloc[self.training_test_threshold:, :]
        else:
            work_df = work_df.iloc[:self.training_test_threshold, :]

        res = pd.concat([parent_df, work_df], axis=1)

        return res

    def get_train_data(self):
        return self._train_data.copy().iloc[:self.training_test_threshold, :]

    def get_test_data(self):
        if not self.has_test_data():
            raise Exception('No test data')
        return self._train_data.copy().iloc[self.training_test_threshold:, :]

    def has_test_data(self):
        return self._test_data is not None

    @staticmethod
    def _stripped_array_representation(arr, limit=5):
        if len(arr) > limit:
            return '{} -> {} and {} others'.format(len(arr),
                                                   ', '.join([str(a) for a in arr[:limit]]),
                                                   len(arr[limit:]))
        else:
            return '{} -> {}'.format(len(arr), ', '.join([str(a) for a in arr]))

    def _feature_overview(self, df, feature_name, limit=10, is_numeric=False):
        null_count = df.isnull()[feature_name].sum()
        all_count = df.isnull()[feature_name].count()
        has_nulls = null_count > 0
        unique_values = df.groupby(feature_name)[feature_name].apply(lambda x: len(x.values))
        unique_values = unique_values.sort_values(ascending=False)
        if is_numeric:
            unique_values = ['{:.2f} ({})'.format(float(f), unique_values[f]) for f in
                             unique_values.index.values]
        else:
            unique_values = ['{} ({})'.format(f, unique_values[f]) for f in
                             unique_values.index.values]
        missing_percent = ' ({:.2f}% missing)'.format(
            null_count * 100 / all_count) if has_nulls else ''
        return '  - {}{}: {}'.format(feature_name, missing_percent,
                                     self._stripped_array_representation(unique_values, limit))

    def overview(self, verbose=False):
        l = 1000000 if verbose else 5
        features_before_preproc = self._stripped_array_representation(self.features, l)
        _, _, _, selected_features = self.build_Xy(
            warnings=False)  # Pre-process does not include dim reduction
        features_after_preproc = self._stripped_array_representation(selected_features, l)
        categorical_features = self._stripped_array_representation(self.categorical_features(), l)
        numerical_features = self._stripped_array_representation(self.numerical_features(), l)
        tr_data = self.preprocess(include_one_hot=False)
        training_data_count = tr_data.shape[0]
        test_data_count = self.get_test_data().shape[0] if self.has_test_data() else 0
        tmp = '\nFeature Source: \'{}\'\n\n' \
              ' o Training samples: {}\n' \
              ' o Test samples: {}\n' \
              ' o Target: \n{}\n\n' \
              ' o Categorical features: {}\n' \
              '{}\n\n' \
              ' o Numerical features: {}\n' \
              '{}\n\n' \
              ' o Before pre-processing: {}\n' \
              ' o After pre-processing: {}\n' \
            .format(self.name,
                    training_data_count,
                    test_data_count,
                    self._feature_overview(self.get_train_data(), self.target, l),
                    categorical_features,
                    '\n'.join(
                        [self._feature_overview(tr_data, f, l) for f in
                         self.categorical_features()]),
                    numerical_features,
                    '\n'.join(
                        [self._feature_overview(tr_data, f, l, is_numeric=True) for f in
                         self.numerical_features()]),
                    features_before_preproc,
                    features_after_preproc)
        return tmp

    def categorical_features(self):
        df = self.preprocess(include_one_hot=False)
        cats = df.select_dtypes(include=[object]).columns.values
        return cats

    def numerical_features(self):
        # TODO This is not quite accurate, handle dates etc in the future
        df = self.preprocess(include_one_hot=False)
        nums = df.select_dtypes(exclude=[object]).columns.values.tolist()
        return nums

    def _unique_values_for(self, feature_name, preprocess=False):
        df = self.get_train_data() if not preprocess else self.preprocess()
        return df[feature_name].unique().values

    def _features_including_parents(self):
        p_features = self.parent._features_including_parents() if self.parent else []
        return [a for a in (p_features + list(self.features)) if a != self.target]

    def _transformations_including_parents(self):
        tfs = self.parent._transformations_including_parents() if self.parent else []
        return tfs + self.transformations

    def __repr__(self):
        tfs = '\n'.join(['  -- {}'.format(a) for a in self._transformations_including_parents()])
        return ' -- FEATURES: {}\n{}\n'.format(', '.join(self._features_including_parents()), tfs)


def VotingBuilder(configs, voting='soft'):
    return VotingClassifier(estimators=[(c.name, c.build_model()) for c in configs], voting=voting)


def StackingBuilder(configs, meta_classifier=LogisticRegression(), use_probas=True,
                    average_probas=False):
    return StackingClassifier(
        classifiers=[c.build_model() for c in configs],
        meta_classifier=meta_classifier,
        use_probas=use_probas,
        average_probas=average_probas)


class Experiment(object):
    def __init__(self, name, cv, cv_shuffle, sc, parallel=True, strategy='best'):
        self.name = name
        self.configs = []
        self.runs = defaultdict(list)
        self.strategy = strategy
        self.features_sources = []
        self.cv = StratifiedKFold(n_splits=cv, shuffle=cv_shuffle)
        self.cv_shuffle = cv_shuffle
        self.sc = sc
        self.parallel = parallel

    def add_estimator(self, classifier_config):
        self.configs.append(classifier_config)

    def add_features(self, features_source):
        self.features_sources.append(features_source)

    def make_features(self, *args, **kwargs):
        fs = Features(*args, **kwargs)
        self.add_features(fs)
        return fs

    @staticmethod
    def _par_grid_search(config, feature_source, cv, sc, f_sel_threshold, strategy, verbose=True):

        f_sel_threshold = feature_source.set_selector_threshold(f_sel_threshold)
        X_train, y_train, feature_names, sel_feature_names = feature_source.build_Xy()
        best_model, best_params = Experiment.grid_search_cv(config, X_train, y_train, cv, sc,
                                                            strategy)
        config.best_params = best_params

        X_test = None
        if feature_source.has_test_data():
            X_test, _, _ = feature_source.build_X(for_test=True)
        results, prediction_results = Experiment.cross_validate(config, X_train, y_train, cv, sc,
                                                                X_test)
        if verbose:
            print(
                'o {}: mean: {:.4f}, std: {:.2f} ({}: {}/{}), best params: {}\n'.format(
                    config.name, np.mean(results), np.std(results), feature_source.name,
                    len(sel_feature_names), len(feature_names), best_params))

        return (config, best_model, best_params, feature_source, f_sel_threshold, results,
                prediction_results)

    def grid_search_all(self, f_sel_thresholds=['default'], verbose=True):
        print('Experiment \'{}\', running grid search...\n'.format(self.name))
        configs = self.configs

        if self.parallel:
            result_arrays = Parallel(n_jobs=max(1, multiprocessing.cpu_count() - 1))(
                delayed(Experiment._par_grid_search)(c, fs, self.cv, self.sc, fst, self.strategy,
                                                     verbose)
                for fs in self.features_sources for fst in f_sel_thresholds for c in configs)
        else:
            result_arrays = [
                Experiment._par_grid_search(c, fs, self.cv, self.sc, fst, self.strategy, verbose)
                for fs in self.features_sources for fst in f_sel_thresholds for c in configs]

        for (config, best_model, best_params, feature_source, fst, results,
             prediction_results) in result_arrays:
            self.config(config.name).best_params = best_params
            self.runs[config.name].append(
                Result(best_model, best_params, results, prediction_results, feature_source.name,
                       fst))

    def config(self, name):
        for c in self.configs:
            if name == c.name:
                return c
        raise Exception('Unknown config: {}'.format(name))

    def non_stacking_configs(self):
        return [a for a in self.configs if not a.stacking]

    def stacking_configs(self):
        return [a for a in self.configs if a.stacking]

    def find_runs(self, config_name, feature_source_name=None):
        if feature_source_name:
            runs = [r for r in self.runs[config_name] if
                    r.feature_source_name == feature_source_name]
        else:
            runs = self.runs[config_name]
        return runs

    def find_best_run(self, config_name, feature_source_name=None):
        runs = self.find_runs(config_name, feature_source_name)
        runs = sorted(runs, key=lambda r: r.mean_score)
        return runs[-1]

    def find_run(self, config_name, feature_source_name=None):
        runs = self.find_runs(config_name, feature_source_name)
        if len(runs) != 1:
            raise Exception('Were expecting exactly one run for: {}, {}'.format(
                config_name,
                feature_source_name))
        return runs[0]

    def find_models(self, config_name, feature_source_name=None):
        return [r.best_model for r in self.find_runs(config_name, feature_source_name)]

    def find_best_model(self, config_name, feature_source_name=None):
        return self.find_best_run(config_name, feature_source_name).best_model

    def find_model(self, config_name, feature_source_name=None):
        return self.find_run(config_name, feature_source_name).best_model

    def find_features(self, features_name):
        for f in self.features_sources:
            if f.name == features_name:
                return f
        return None

    @staticmethod
    def grid_search_cv(config, X_train, y_train, cv, sc, strategy='best'):

        the_model = config.build_model()

        if not config.grid_params:
            return the_model, config.best_params

        serial = True
        grid_obj = GridSearchCV(the_model, config.grid_params, scoring=sc, cv=cv,
                                n_jobs=1 if serial else 8)
        grid_obj = grid_obj.fit(X_train, y_train)

        if strategy == 'best':
            return grid_obj.best_estimator_, grid_obj.best_params_
        else:
            raise Exception('Implement worst, random etc.')

    def overview(self, verbose=0):
        print('\nExperiment: \'{}\'\n{}'.format(self.name, '\n\n'.join(
            [fs.overview(verbose) for fs in self.features_sources])))

    def plot_cv_runs(self, figsize=7):
        results_df = pd.DataFrame()
        sorted_configs = self.configs  # in case we want a different sorting
        for c in sorted_configs:
            fs_names = defaultdict(list)
            for r in self.runs[c.name]:
                fs_names[r.feature_source_name] = r.results
            for key, values in fs_names.items():
                cfg_df = pd.DataFrame()
                cfg_df['CV Score'] = values
                cfg_df['Features'] = [key] * len(values)
                cfg_df['Config'] = [c.name] * len(values)
                results_df = pd.concat([results_df, cfg_df])
        g = sns.factorplot(x='Features', y='CV Score', hue='Config', data=results_df, kind='violin',
                           size=figsize, legend_out=True)
        g.set_xticklabels(rotation=25)
        title = "Accuracy distribution over {} CV runs, cv shuffled: {}\n".format(self.cv.n_splits,
                                                                                  self.cv_shuffle)
        plt.title(title)

    def plot_f_sel_learning_curve(self):
        results_df = pd.DataFrame()
        sorted_configs = self.configs  # in case we want a different sorting
        for c in sorted_configs:
            fs_names = defaultdict(list)
            for r in self.runs[c.name]:
                fs_names[r.key()] = r.results
            for key, values in fs_names.items():
                cfg_df = pd.DataFrame()
                cfg_df['CV Score'] = values
                cfg_df['Features'] = [key] * len(values)
                cfg_df['Config'] = [c.name] * len(values)
                results_df = pd.concat([results_df, cfg_df])
        g = sns.factorplot(x='Features', y='CV Score', hue='Config', data=results_df, size=7,
                           legend_out=True)
        g.set_xticklabels(rotation=25)
        plt.title("Accuracy vs feature selection threshold\n")

    def plot_feature_importance(self, limit=15, figsize=7):
        importance_df = pd.DataFrame()
        for config_name, runs in self.runs.items():
            for run in runs:
                model = run.best_model
                features = self.find_features(run.feature_source_name)
                X, y, f_names, selected_f_names = features.build_Xy()
                model.fit(X, y)
                if hasattr(model, 'coef_'):
                    imps = model.coef_[0]  # TODO multiclass?
                elif hasattr(model, 'feature_importances_'):
                    imps = model.feature_importances_
                else:
                    imps = [0.] * len(selected_f_names)
                if max(imps) > 1:
                    print(' !!! WARNING !!! {} was scaled to 0.5 for better visualization'.format(
                        config_name))
                    imps = np.asarray(imps) / (2 * np.max(np.asarray(imps), axis=0))
                features_source_name = run.feature_source_name if not limit else '{} (limit: {})'.format(
                    run.feature_source_name, limit)
                local = pd.DataFrame({'Feature': selected_f_names, 'Importance': imps,
                                      'Estimator': [config_name] * len(selected_f_names),
                                      'Features': [features_source_name] * len(
                                          selected_f_names)})
                if limit:
                    local = local.sort_values(by='Importance', ascending=False).iloc[:limit, :]
                importance_df = pd.concat([importance_df, local])
        g = sns.factorplot(x='Feature', y='Importance', col='Features', hue='Estimator',
                           data=importance_df.sort_values(by=['Importance'], ascending=False),
                           kind='bar', size=figsize)
        g.set_xticklabels(rotation=75)

    def plot_correlations(self, top_n=15, figsize=7):
        for fs in self.features_sources:
            plt.figure(figsize=(1.1 * figsize, figsize))
            df = fs.preprocess()
            X, y, _, _ = fs.build_Xy()
            df = pd.concat([df, pd.DataFrame(y, columns=[fs.target])], axis=1)
            cols = df.corr().nlargest(top_n, fs.target)[fs.target].index
            cm = np.corrcoef(df[cols].values.T)
            sns.set(font_scale=1.2)
            g = sns.heatmap(cm, cbar=True, annot=True, fmt='.2f', annot_kws={'size': 10},
                            yticklabels=cols.values, xticklabels=cols.values)
            plt.yticks(rotation=0)
            plt.xticks(rotation=90)
            plt.title('\'{}\', top {} highest correlations\n'.format(fs.name, top_n))

    def show_null_stats(self, preprocess=False):
        for fs in self.features_sources:
            train_data = fs.preprocess() if preprocess else fs.get_train_data()
            if fs.target in train_data.columns:
                train_data = train_data.drop([fs.target], axis=1)
            self.show_null_stats_for(train_data, '{} Training / {}'.format(self.name, fs.name))

            if fs.has_test_data():
                test_data = fs.preprocess(for_test=True) if preprocess else fs.get_test_data()
                if fs.target in test_data.columns:
                    test_data = test_data.drop(fs.target, axis=1)
                self.show_null_stats_for(test_data, '{} Test / {}'.format(self.name, fs.name))

    @staticmethod
    def show_null_stats_for(df, name=None):
        nulls = pd.DataFrame({'Empty': df.isnull().sum()})
        counts = pd.DataFrame({'Empty': df.isnull().count()})
        if nulls[nulls['Empty'] > 0].empty:
            print('No null values found in \'{}\''.format(name))
            return
        nulls = nulls[nulls['Empty'] > 0].sort_values(by='Empty', ascending=False).transpose()
        counts = counts.transpose()[nulls.columns]
        plt.figure(figsize=(9, 6))
        sns.barplot(data=counts.iloc[:len(nulls.index)], palette=['lightgray'])
        ax = sns.barplot(data=nulls, palette=['darkred'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        postfix = '' if not name else ' ({})'.format(name)
        plt.title('Null / NA Counts{}\n'.format(postfix))
        plt.xticks(rotation=75)
        # plt.yticks([])
        ax.grid(False)
        p0 = ax.patches[-len(nulls.columns)]
        max_height = counts.iloc[0, :][0]
        ax.text(p0.get_x() + 0.4, max_height + 15, '{:g}'.format(max_height), ha="center",
                va='bottom', fontsize=14, color='black')
        for p in ax.patches[-len(nulls.columns):]:
            height = p.get_height()
            if height > 0:
                ax.text(p.get_x() + p.get_width() / 2., height + 20, '{:g}'.format(height),
                        ha="center", va='bottom', fontsize=14, color='darkred')

    @staticmethod
    def _brute_force_ensemble_combo(combo, voting, X_train, y_train):
        cls = VotingBuilder(list(combo)) if voting else StackingBuilder(list(combo))
        results = Experiment.cross_validate_model(cls, X_train, y_train)
        return list(combo), results

    def brute_force_ensemble(self, X_train, y_train, voting=True, use_ensembles_on_lvl1=False):
        configs_to_use = self.configs if use_ensembles_on_lvl1 else self.non_stacking_configs()
        all_combos = [combo for combo_size in range(3, len(configs_to_use) + 1, 2) for combo in
                      combinations(configs_to_use, combo_size)]
        n = len(all_combos)
        print(' -- Brute forcing {} combos for the {} classifier...'.format(
            n, 'voting' if voting else 'stacking'))

        res_combos = Parallel(n_jobs=8)(delayed(Experiment._brute_force_ensemble_combo)(
            c, voting, X_train, y_train) for c in all_combos)

        res_combos = sorted(res_combos, key=lambda x: np.mean(x[1]), reverse=True)
        print(' -- Brute forced the following combos for the {} classifier:\n'.format(
            'voting' if voting else 'stacking'
        ))
        for combo in res_combos[:10]:
            print('    o {}: {:.4f} \u00B1 {:.2f}'.format(', '.join([cb.name for cb in combo[0]]),
                                                          np.mean(combo[1]), np.std(combo[1])))

    @staticmethod
    def cross_validate(config, X_train, y_train, cv, sc, X_test=None):
        return Experiment.cross_validate_model(config.build_model(), X_train, y_train, cv, sc,
                                               X_test)

    @staticmethod
    def cross_validate_model(model, X_train, y_train, cv, sc, X_test=None):
        _scorer = get_scorer(sc)
        our_results = []
        prediction_results = []
        for train_index, val_index in cv.split(X_train, y_train):
            X_train_cv, X_val_cv = X_train[train_index], X_train[val_index]
            y_train_cv, y_val_cv = y_train[train_index], y_train[val_index]
            model.fit(X_train_cv, y_train_cv)
            the_score = _scorer(model, X_val_cv, y_val_cv)
            our_results.append(the_score)
            X_test = X_test if X_test is not None else X_val_cv
            y_pred = model.predict(X_test)
            prediction_results.append(y_pred)
        # their_results = cross_val_score(model, X_train, y_train, scoring=sc, cv=cv)
        # print('{} \n {} \n\n'.format(their_results, our_results))
        return our_results, prediction_results
