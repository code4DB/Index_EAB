# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: optimizer_model
# @Author: Wei Zhou
# @Time: 2023/10/6 19:55


from sklearn.linear_model import LinearRegression


class Optimizer:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def estimate(self, x_test):
        return self.model.predict(x_test)
