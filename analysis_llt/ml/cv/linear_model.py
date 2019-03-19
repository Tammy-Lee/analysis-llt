# -*- coding: utf-8 -*-
"""
create on 2019-03-20 04:20

author @lilia
"""
from sklearn.linear_model import LogisticRegression

from analysis_llt.ml.cv.base import BaseCV


class LogisticRegressionCVLLT(BaseCV):
    def __init__(self, penalty='l2', C=1.0, cv=None, class_weight=None, random_state=None, solver='warn', verbose=0,
                 **model_params):
        super(LogisticRegressionCVLLT, self).__init__(cv=cv, random_state=random_state, verbose=verbose, **model_params)
        self.C = C
        self.penalty = penalty
        self.class_weight = class_weight
        self.solver = solver

    def build_model(self):
        lr = LogisticRegression(penalty=self.penalty, C=self.C, class_weight=self.class_weight, solver=self.solver,
                                random_state=self.random_state, **self.model_params)
        return lr
