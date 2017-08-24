import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler


class Transformation(object):
    def __init__(self, feature_names, from_=None):
        self.feature_names = feature_names
        self.from_ = feature_names if from_ is None else from_
        self.std_scalers = {}
        if len(self.from_) != len(self.feature_names):
            raise Exception('Output feature names should be as many as in feature names')

    def apply_to(self, source_df):
        source_df = self._internal_apply_to(source_df)
        for feature in [x for x in self.feature_names if x in source_df.columns]:
            null_count = source_df[feature].isnull().sum()
            if null_count > 0:
                print('\n!!! WARNING !!! {} has {} NULL/NA values after applying {} '
                      'transform\n'.format(feature, null_count, self.friendly_name()))
        return source_df

    def _internal_apply_to(self, source_df):
        return source_df

    def friendly_name(self):
        return self.__class__.__name__.replace('Transformation', '').lower()

    def __repr__(self):
        return 'TRANSFORM: {} -> {}'.format(', '.join(self.from_), self.friendly_name())


class OneHotTransformation(Transformation):
    def __init__(self, feature_names, from_=None, drop_first=False):
        super().__init__(feature_names, from_)
        self.drop_first = drop_first

    def _internal_apply_to(self, source_df):
        return pd.get_dummies(source_df, columns=self.feature_names, drop_first=self.drop_first)


class MethodTransformation(Transformation):
    def __init__(self, feature_names, method_handle, from_=None):
        super().__init__(feature_names, from_)
        self.method_handle = method_handle

    def _internal_apply_to(self, source_df):
        for idx, feature in enumerate(self.feature_names):
            from_feature = self.from_[idx]
            if self.method_handle:
                source_df[feature] = self.method_handle(source_df)
            else:
                # Just duplicate
                source_df[feature] = source_df[from_feature]
        return source_df


class MathTransformation(MethodTransformation):
    @staticmethod
    def mul(x, y):
        return x * y

    @staticmethod
    def sum(x, y):
        return x + y

    @staticmethod
    def div(x, y):
        return x / y

    def _internal_apply_to(self, source_df):
        for idx, feature in enumerate(self.feature_names):
            to_multiply = self.from_[idx].split(',')
            assert len(to_multiply) >= 2
            starter = source_df[to_multiply[0]]
            for n in [a.strip() for a in to_multiply[1:]]:
                if n in source_df.columns:
                    starter = self.method_handle(starter, source_df[n])
                else:
                    # TODO HUGE WARNING HERE - use an eval library here
                    starter = self.method_handle(starter, eval(n))
            source_df[feature] = starter
        return source_df


class MulTransformation(MathTransformation):
    def __init__(self, feature_names, from_=None):
        super().__init__(feature_names, MathTransformation.mul, from_)


class SumTransformation(MathTransformation):
    def __init__(self, feature_names, from_=None):
        super().__init__(feature_names, MathTransformation.sum, from_)


class DivTransformation(MathTransformation):
    def __init__(self, feature_names, from_=None):
        super().__init__(feature_names, MathTransformation.div, from_)


class MappingTransformation(Transformation):
    ALL_OTHERS = '__ALL_OTHERS__'

    @staticmethod
    def all_others_mapping(x, mapping):
        default_value = mapping.get('_others')
        # we need this in case of  --> 0 mappings which evaluate to False
        if default_value is None:
            default_value = mapping[MappingTransformation.ALL_OTHERS]
        return mapping.get(x, default_value)

    def __init__(self, feature_names, mapping, from_=None):
        super().__init__(feature_names, from_)
        self.mapping = mapping

    def _internal_apply_to(self, source_df):
        for idx, feature in enumerate(self.feature_names):
            from_feature = self.from_[idx]
            if MappingTransformation.ALL_OTHERS in self.mapping or '_others' in self.mapping:
                source_df[feature] = source_df[from_feature].apply(
                    MappingTransformation.all_others_mapping, args=(self.mapping,))
            else:
                source_df[feature] = source_df[from_feature].map(self.mapping)
        return source_df


class DiscreteTransformation(Transformation):
    def __init__(self, feature_names, values_range=None, q=None, from_=None):
        super().__init__(feature_names, from_)
        self.values_range = values_range
        self.q = q

    def _internal_apply_to(self, source_df):
        for idx, feature in enumerate(self.feature_names):
            from_feature = self.from_[idx]
            if self.values_range is not None:
                labels = [str(r) for r in self.values_range]
                source_df[feature] = pd.cut(source_df[from_feature],
                                            bins=self.values_range + [sys.float_info.max],
                                            labels=labels)
            else:
                labels = range(self.q)
                source_df[feature] = pd.qcut(source_df[from_feature], q=self.q, labels=labels)
        return source_df


class LogTransformation(Transformation):
    def _internal_apply_to(self, source_df):
        for idx, feature in enumerate(self.feature_names):
            from_feature = self.from_[idx]
            source_df[feature] = np.log1p(source_df[from_feature] + 1)
        return source_df


class FillnaTransformation(Transformation):
    def __init__(self, feature_names, strategy='mean', value=None, from_=None):
        super().__init__(feature_names, from_)
        self.strategy = strategy
        self.value = value

    def _internal_apply_to(self, source_df):
        for idx, feature in enumerate(self.feature_names):
            from_feature = self.from_[idx]
            if self.strategy == 'mean':
                source_df[feature] = source_df[from_feature].fillna(source_df[from_feature].mean())
            elif self.strategy == 'median':
                source_df[feature] = source_df[from_feature].fillna(
                    source_df[from_feature].median())
            elif self.strategy == 'value':
                source_df[feature] = source_df[from_feature].fillna(self.value)
            else:
                raise Exception('Unknown NA fill strategy')
        return source_df


class StdTransformation(Transformation):
    def _internal_apply_to(self, source_df):
        for idx, feature in enumerate(self.feature_names):
            from_feature = self.from_[idx]
            if source_df[feature].isnull().sum() > 0:
                print(' !!! WARNING !!! {} has null values, replacing with mean'.format(feature))
                source_df[feature] = source_df[from_feature].fillna(source_df[from_feature].mean())

            if not self.std_scalers.get(feature):
                # Create and fit it first time (training set) then apply the same to test
                scaler = StandardScaler()
                scaler.fit(source_df[from_feature].values.reshape(-1, 1))
                self.std_scalers[feature] = scaler

            source_df[feature] = self.std_scalers[feature].transform(
                source_df[from_feature].values.reshape(-1, 1))
        return source_df


T_MAP = {
    'onehot': OneHotTransformation,
    'method': MethodTransformation,
    'log': LogTransformation,
    'std': StdTransformation,
    'map': MappingTransformation,
    'discretize': DiscreteTransformation,
    'fillna': FillnaTransformation,
    '*': MulTransformation,
    '+': SumTransformation,
    '/': DivTransformation,
}
