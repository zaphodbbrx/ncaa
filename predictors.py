import random
from abc import ABC, abstractmethod

import lightgbm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV


class BasePredictor(ABC):

    def __init__(self):
        self.is_trained = False

    @abstractmethod
    def _train_fn(self, X, y):
        pass

    @abstractmethod
    def _predict_fn(self, x):
        pass

    def train(self, X, y):
        self._train_fn(X, y)
        self.is_trained = True

    def __call__(self, x):
        if self.is_trained:
            return self._predict_fn(x)
        else:
            raise ValueError('model is not trained')


class DummyPredictor(BasePredictor):

    def _train_fn(self, X, y):
        pass

    def _predict_fn(self, x):
        return random.random()


class LightGBMPredictor(BasePredictor):

    def __init__(self, categorical):
        super().__init__()
        self.categorical = categorical

    def _train_fn(self, X, y):
        for c in self.categorical:
            X[c] = X[c].astype('category')
        parameters = {'n_estimators': 10000,
                 'num_leaves': 32,
                 'min_child_weight': 0.034,
                 'feature_fraction': 0.379,
                 'bagging_fraction': 0.418,
                 'min_data_in_leaf': 106,
                 'objective': 'binary',
                 'max_depth': -1,
                 'learning_rate': 0.0068,
                 "boosting_type": "gbdt",
                 # "bagging_seed": 11,
                 "metric": 'binary_logloss',
                 "verbosity": 10,
                 'reg_alpha': 0.3899,
                 'reg_lambda': 0.648,
                 'random_state': 47,
                 'task': 'train', 'nthread': -1,
                 'verbose': 1000,
                 'early_stopping_rounds': 500,
                 'eval_metric': 'binary_logloss'
                 }
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
        train_dataset = lightgbm.Dataset(X_train, label=y_train, categorical_feature=self.categorical)
        val_dataset = lightgbm.Dataset(X_val, label=y_val, categorical_feature=self.categorical)
        self.model = lightgbm.train(parameters,
                                    train_dataset,
                                    # init_model=lightgbm.LGBMClassifier(),
                                    valid_sets=val_dataset,
                                    num_boost_round=9000,
                                    early_stopping_rounds=100)
        # self.model = lightgbm.LGBMClassifier(
        #     n_estimators=9000,
        #     objective='binary',
        #     early_stopping_rounds=1000,
        # )
        # self.model.fit(
        #     X_train,
        #     y_train,
        #     eval_set=[(X_val, y_val)],
        #     eval_metric='binary_logloss',
        # )

    def _predict_fn(self, x):
        for c in self.categorical:
            x[c] = x[c].astype('category')
        return self.model.predict(x)


class SklearnPredictor(BasePredictor):

    def _train_fn(self, X, y):

        self.model = LogisticRegressionCV(class_weight='balanced')
        self.model.fit(X, y)

    def _predict_fn(self, x):
        return self.model.predict(x)
