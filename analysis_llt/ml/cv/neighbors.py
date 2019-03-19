# -*- coding: utf-8 -*-
"""
create on 2019-03-20 04:17

author @lilia
"""
from sklearn.neighbors import KNeighborsClassifier

from analysis_llt.ml.cv.base import BaseCV


class KNNCV(BaseCV):
    fit_predict_proba_ = False

    def __init__(self, n_neighbors=5, cv=None, random_state=None, verbose=0, **model_params):
        super(KNNCV, self).__init__(cv=cv, random_state=random_state, verbose=verbose, **model_params)
        self.n_neighbors = n_neighbors
        if 'predict_proba' in model_params:
            warnings.warn("SVC does not have predict_proba function")

    def build_model(self):
        knn = KNeighborsClassifier(n_neighbors=self.n_neighbors, **self.model_params)
        return knn
