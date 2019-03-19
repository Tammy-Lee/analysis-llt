# -*- coding: utf-8 -*-
"""
create on 2019-03-20 04:16

author @lilia
"""
from sklearn.ensemble import RandomForestClassifier

from analysis_llt.ml.cv.base import BaseCV


class RandomForestClassifierCV(BaseCV):
    def __init__(self, n_estimators='warn', criterion='gini', max_depth=None, min_samples_split=1.0, cv=None,
                 random_state=None, verbose=0, **model_params):
        super(RandomForestClassifierCV, self).__init__(cv=cv, random_state=random_state, verbose=verbose,
                                                       **model_params)
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def build_model(self):
        forest = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion,
                                        max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                        random_state=self.random_state, **self.model_params)
        return forest
