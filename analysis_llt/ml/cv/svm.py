# -*- coding: utf-8 -*-
"""
create on 2019-03-20 04:12

author @lilia
"""
import warnings

from sklearn.svm import SVC

from analysis_llt.ml.cv.base import BaseCV


class SVCCV(BaseCV):
    fit_predict_proba_ = False

    def __init__(self, C=1.0, cv=None, random_state=None, verbose=0, **model_params):
        super(SVCCV, self).__init__(cv=cv, random_state=random_state, verbose=verbose, **model_params)
        self.C = C
        if 'predict_proba' in model_params:
            warnings.warn("SVC does not have predict_proba function")

    def build_model(self):
        svc = SVC(C=self.C, random_state=self.random_state, **self.model_params)
        return svc
