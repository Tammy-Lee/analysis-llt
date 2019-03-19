# -*- coding: utf-8 -*-
"""
create on 2019-03-20 04:19

author @lilia
"""
from sklearn.neural_network import MLPClassifier

from analysis_llt.ml.cv.base import BaseCV


class MLPClassifierCV(BaseCV):
    def __init__(self, hidden_layer_sizes=(100,), learning_rate_init=0.001, cv=None, random_state=None, verbose=0,
                 **model_params):
        super(MLPClassifierCV, self).__init__(cv=cv, random_state=random_state, verbose=verbose, **model_params)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate_init = learning_rate_init

    def build_model(self):
        mlp = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, random_state=self.random_state,
                            learning_rate_init=self.learning_rate_init, **self.model_params)
        return mlp
