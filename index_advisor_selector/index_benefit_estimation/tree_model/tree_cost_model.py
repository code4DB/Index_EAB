# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: tree_cost_model
# @Author: Wei Zhou
# @Time: 2022/8/11 9:41

# https://stackoverflow.com/questions/69137780/provide-additional-custom-metric-to-lightgbm-for-early-stopping

import joblib

import lightgbm as lgb
import xgboost as xgb

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from index_advisor_selector.index_benefit_estimation.tree_model.tree_cost_utils.tree_cost_loss import xgb_QError, lgb_QError, xgb_MSE, lgb_MSE


class XGBoost:
    def __init__(self, nthread=10, path=None):
        if path is not None:
            self.model = xgb.Booster(model_file=path)
        else:
            self.model = xgb.Booster({"nthread": nthread})

    def train(self, x_train, y_train, num_rounds=10, params=None):
        if params is None:
            params = {"max_depth": 5, "eta": 0.1, "booster": "gbtree",
                      "objective": "reg:logistic", "seed": 666}
        train_len = int(0.8 * len(x_train))
        xgb_train = xgb.DMatrix(x_train[:train_len], label=y_train[:train_len])
        xgb_valid = xgb.DMatrix(x_train[train_len:], label=y_train[train_len:])
        evallist = [(xgb_train, "train"), (xgb_valid, "test")]
        self.model = xgb.train(params, xgb_train, num_rounds, evallist,
                               early_stopping_rounds=150, feval=xgb_MSE)

    def save_model(self, path):
        self.model.save_model(path + ".xgb.model")

    def load_model(self, path):
        self.model.load_model(path + ".xgb.model")

    def estimate(self, test_data):
        dtest = xgb.DMatrix(test_data)
        return self.model.predict(dtest)


class LightGBM:
    def __init__(self, path=None):
        if path is not None:
            self.model = lgb.Booster(model_file=path)
        else:
            self.model = None

    def train(self, x_train, y_train, num_rounds=20, params=None):
        if params is None:
            params = {
                "task": "train",
                "boosting_type": "gbdt",  # 设置提升类型
                "objective": "regression",  # 目标函数
                "metric": {"l2", "auc"},  # 评估函数
                "num_leaves": 31,  # 叶子节点数
                "learning_rate": 0.05,  # 学习速率
                "feature_fraction": 0.9,  # 建树的特征选择比例
                "bagging_fraction": 0.8,  # 建树的样本采样比例
                "bagging_freq": 5,  # k 意味着每 k 次迭代执行bagging
                "verbose": 1,  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
                "seed": 666
            }

        train_len = int(0.8 * len(x_train))
        lgb_train = lgb.Dataset(x_train[:train_len], y_train[:train_len])
        # If this is Dataset for validation, training data should be used as reference.
        lgb_eval = lgb.Dataset(x_train[train_len:], y_train[train_len:], reference=lgb_train)

        # self.model = lgb.train(params, lgb_train, num_boost_round=num_rounds,
        #                        valid_sets=lgb_eval, early_stopping_rounds=5)
        self.model = lgb.train(params, lgb_train,
                               num_boost_round=num_rounds, valid_sets=(lgb_train, lgb_eval),
                               early_stopping_rounds=50, feval=lgb_MSE)

    def save_model(self, path):
        self.model.save_model(path + ".lgb.model")

    def load_model(self, path):
        self.model = lgb.Booster(model_file=path)

    def estimate(self, data):
        # dtest = lgb.Dataset(data)
        # return self.model.predict(dtest)
        return self.model.predict(data)


class RandomForest:
    def __init__(self, task_type="reg", path=None):
        self.task_type = task_type
        if path is not None:
            self.model = joblib.load(path)
        else:
            if task_type == "reg":
                self.model = RandomForestRegressor()
            elif task_type == "cla":
                self.model = RandomForestClassifier()

    def train(self, x_train, y_train, params=None):
        if params is None:
            params = {
                "task": "train",
                "boosting_type": "gbdt",  # 设置提升类型
                "objective": "regression",  # 目标函数
                "metric": {"l2", "auc"},  # 评估函数
                "num_leaves": 31,  # 叶子节点数
                "learning_rate": 0.05,  # 学习速率
                "feature_fraction": 0.9,  # 建树的特征选择比例
                "bagging_fraction": 0.8,  # 建树的样本采样比例
                "bagging_freq": 5,  # k 意味着每 k 次迭代执行bagging
                "verbose": 1,  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
                "seed": 666
            }
        self.model.set_params(**params)
        # rfr = RandomForestRegressor(n_estimators=250, random_state=666, max_depth=8)
        self.model.fit(x_train, y_train)

    def save_model(self, path):
        joblib.dump(self.model, path + ".rf.joblib")
        # self.model.save_model(path + ".rf.model")

    def load_model(self, path):
        self.model = joblib.load(path)

    def estimate(self, data):
        return self.model.predict(data)
