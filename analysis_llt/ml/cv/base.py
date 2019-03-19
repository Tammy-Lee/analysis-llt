# -*- coding: utf-8 -*-
"""
create on 2019-03-20 04:07

author @lilia
"""
from sklearn import metrics
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold

SCORES = {
    "roc_auc": metrics.roc_auc_score,
    "f1_score": metrics.f1_score,
    "precision": metrics.precision_score,
    "recall": metrics.recall_score,
    "accuracy": metrics.accuracy_score
}


class BaseCV(BaseEstimator):
    fit_predict_proba_ = True

    def __init__(self, cv=None, random_state=None, verbose=0, **model_params):
        """"""
        self.cv = cv
        self.random_state = random_state
        self.verbose = verbose
        self.model_params = model_params
        self.models = []
        self.cv_y_trues = []
        self.cv_y_preds = []
        if 'predict_proba' in model_params:
            self.fit_predict_proba_ = model_params.pop('predict_proba')

    def build_model(self):
        raise NotImplementedError("Remember to implement this method!")

    def fit(self, X, y, sample_weight=None):
        kfold = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        for i, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            train_x, train_y = X[train_idx], y[train_idx]
            val_x, val_y = X[val_idx], y[val_idx]
            model = self.build_model()
            model.fit(train_x, train_y)
            if self.fit_predict_proba_:
                cv_y_pred = model.predict_proba(val_x)[:, 1]
            else:
                cv_y_pred = model.predict(val_x)
            self.models.append(model)
            self.cv_y_trues.append(val_y)
            self.cv_y_preds.append(cv_y_pred)
        return self

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) > threshold).astype(int)

    def predict_proba(self, X):
        y_preds = []
        for model in self.models:
            if self.fit_predict_proba_:
                y_pred = model.predict_proba(X)[:, 1]
            else:
                y_pred = model.predict(X)
            y_preds.append(y_pred)
        return np.mean(y_preds, axis=0)

    def report_cv(self, y_true=None, y_pred=None, scoring=('roc_auc', 'precision', 'recall', 'f1_score', 'accuracy'),
                  threshold=0.5):
        scores = []
        for y_pred, y_true in zip(self.cv_y_preds, self.cv_y_trues):
            scores.append(self.report(y_true=y_true, y_pred=y_pred, scoring=scoring, threshold=threshold))
        return pd.DataFrame(scores).mean().to_dict()

    def report(self, y_true=None, y_pred=None, scoring=('roc_auc', 'precision', 'recall', 'f1_score', 'accuracy'),
               threshold=0.5, errors="ignore"):
        report_output = {}
        for score in scoring:
            try:
                report_output[score] = SCORES[score](y_true, y_pred)
            except ValueError as e:
                if errors == 'print':
                    print(e)
                report_output[score] = SCORES[score](y_true, (y_pred > threshold).astype(int))
        return report_output
