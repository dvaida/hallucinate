import os
import pandas as pd
import numpy as np


class Kaggle(object):
    def __init__(self, experiment, index_feature, target_feature=None):
        self.experiment = experiment
        self.index_feature = index_feature
        self.target_feature = target_feature

    def create_submissions(self, estimator_config_name=None, features_name=None,
                           predictions_to_int=False):
        print('\n -- Creating submissions...\n')
        os.makedirs('submissions', exist_ok=True)

        all_predictions = {}
        all_cv_runs = []  # for voting on prediction results from ALL runs
        target_feature = self.target_feature  # needed for voting prediction
        latest_features = None  # needed for voting prediction

        for config in self.experiment.configs:
            if estimator_config_name and config.name != estimator_config_name:
                continue

            config_cv_runs = []  # for voting on prediction results from all this config runs

            runs = self.experiment.find_runs(config.name)
            for run in runs:
                if features_name and features_name != run.feature_source_name:
                    continue

                model = run.best_model
                config_cv_runs.append(run.prediction_results)
                all_cv_runs.append(run.prediction_results)
                submission_key = run.key()
                features = self.experiment.find_features(run.feature_source_name)
                if run.f_sel_threshold:
                    features.set_selector_threshold(run.f_sel_threshold)
                latest_features = features

                X, y, _, _ = features.build_Xy()
                model.fit(X, y)
                X_test, _, _ = features.build_X(for_test=True)
                predictions = model.predict(X_test)

                if predictions_to_int:
                    predictions = np.asarray(predictions).astype(int)
                all_predictions['{}_{}'.format(config.name, submission_key)] = predictions
                target_feature = self.target_feature if self.target_feature else features.target
                submission = pd.concat([features.get_test_data()[self.index_feature],
                                        pd.Series(predictions.astype(int), name=target_feature)],
                                       axis=1)
                out_file_name = 'submissions/submission_{}_{}.csv'.format(config.name,
                                                                          submission_key)
                submission.to_csv(out_file_name, index=False)
                print(' -- Submission written to: \'{}\''.format(out_file_name))

            self._create_cv_voting_submission(config_cv_runs, config.name, latest_features,
                                              target_feature, all_predictions)

        self._create_cv_voting_submission(all_cv_runs, None, latest_features,
                                          target_feature, all_predictions)

        print('\n -- Done creating submissions.\n')
        return pd.DataFrame(all_predictions)

    def _create_cv_voting_submission(self, cv_run_predictions, config_name, latest_features,
                                     target_feature, all_predictions):
        config_name = config_name if config_name else 'GLB'  # global if None
        # Write a all CV runs voting prediction result
        vot = np.asarray(cv_run_predictions)
        vot = vot.reshape((vot.shape[0] * vot.shape[1], vot.shape[2]))
        for strategy in ['majority', 'unanimity']:
            predictions = self.vote_predict(vot, strategy)
            voting_out_name = '{}_CV_VOT_{}'.format(config_name, strategy)
            all_predictions[voting_out_name] = predictions
            submission = pd.concat([latest_features.get_test_data()[self.index_feature],
                                    pd.Series(predictions, name=target_feature)], axis=1)
            voting_out_file_name = 'submissions/submission_{}.csv'.format(voting_out_name)
            submission.to_csv(voting_out_file_name, index=False)
            print(' -- Submission written to: \'{}\''.format(voting_out_file_name))

    @staticmethod
    def vote_predict(predictions, strategy='majority', cast_to_int=True):
        '''
        Expects a list of prediction lists and it will apply the voting on each columns (thus numpy along axis 0),
        aka if the shape is (5, 10), e.g. 5 rows with 10 predictions each, it will create a (10, ) prediction voting result
        '''
        predictions = np.asarray(predictions)
        if strategy == 'majority':
            res = np.rint(np.mean(predictions, axis=0))
            return res.astype(int) if cast_to_int else res
        elif strategy == 'unanimity':
            # float casting to int resolves with rounding down to 0
            # besides unanimity voting with floats doesn't make much sense
            return np.mean(predictions, axis=0).astype(int)
        elif strategy == 'average':
            return np.mean(predictions, axis=0)
        else:
            raise Exception('Unknown voting strategy: \'{}\''.format(strategy))
