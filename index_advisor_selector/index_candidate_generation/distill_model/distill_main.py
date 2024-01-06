# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: distill_main
# @Author: Wei Zhou
# @Time: 2022/8/11 10:58

import os
import json
import random
import numpy as np
import logging

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

from index_advisor_selector.index_candidate_generation.distill_model.distill_utils import distill_com
from index_advisor_selector.index_candidate_generation.distill_model.distill_utils.distill_loss import cal_mape, QError

from index_advisor_selector.index_candidate_generation.distill_model.distill_dataset import PlanPairDataset, normalize, unnormalize
from index_advisor_selector.index_candidate_generation.distill_model.distill_model import XGBoost, LightGBM, RandomForest

# : 1. get the params.
parser = distill_com.get_parser()
args = parser.parse_args()

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# : 3. create the directory to store the `exp_res`.
assert not os.path.exists(os.path.dirname(args.logdir.format(args.exp_id))), \
    f"`{os.path.dirname(args.logdir.format(args.exp_id))}` dir existed!"
os.makedirs(os.path.dirname(args.logdir.format(args.exp_id)))
os.makedirs(os.path.dirname(args.model_save.format(args.exp_id, 0)))
os.makedirs(os.path.dirname(args.data_save.format(args.exp_id, 0)))

distill_com.set_logger(args.runlog.format(args.exp_id))
logging.info("Start Index Filter Experiment.")

logging.info(f"Create the directory `{os.path.dirname(args.logdir.format(args.exp_id))}` to save experiment result.")

# specify the path to store the exp_res of `logdir` of the tensorboard.
distill_com.summary_writer = SummaryWriter(args.logdir.format(args.exp_id))
distill_com.summary_writer.add_text(
    "parameters",
    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    0
)
logging.info(f"Set the tensorboard logdir = `{args.logdir.format(args.exp_id)}`.")

# : 4. set the torch random_seed.
# Sets the seed for generating random numbers.
# Returns a `torch.Generator` object.
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
logging.info(f"Set the random seed = `{args.seed}`.")

# : 5. load the training data.
# ["utility", "query_shape", "index_shape", "physical_operator"]

# args.train_data_load = "/data1/wz/index/index_eab/eab_other/distill_model/data/tpch/tree_tpch_cost_data_tgt_train.json"
with open(args.train_data_load, "r") as rf:
    train_data = json.load(rf)
# args.valid_data_load = "/data1/wz/index/index_eab/eab_other/distill_model/data/tpch/tree_tpch_cost_data_tgt_valid.json"
with open(args.valid_data_load, "r") as rf:
    valid_data = json.load(rf)

train_data = [dat for dat in train_data if dat["label act"] != 0.]
valid_data = [dat for dat in valid_data if dat["label act"] != 0.]

logging.info(f"Load the train data from `{args.train_data_load}` ({len(train_data)}).")
logging.info(f"Load the valid data from `{args.valid_data_load}` ({len(valid_data)}).")

train_set = PlanPairDataset(train_data)
valid_set = PlanPairDataset(valid_data)

# train_set_, valid_set_ = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
# train_set, valid_set = list(train_set_), list(valid_set_)
# train_set.dataset[0]

X_train = [sample[0] for sample in train_set]
y_train = [sample[1] for sample in train_set]
X_valid = [sample[0] for sample in valid_set]
y_valid = [sample[1] for sample in valid_set]

# Compute the minimum and maximum to be used for later scaling
# "cost", "row", "cost_row", "tra_order", "seq_ind"
# # ["utility", "query_shape", "index_shape", "physical_operator"]
if True:
    """
    Transform features by scaling each feature to a given range.
    This estimator scales and translates each feature individually such
    that it is in the given range on the training set, e.g. between
    zero and one.

    The transformation is given by::
        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min
    where min, max = feature_range.

    This transformation is often used as an alternative to zero mean,
    unit variance scaling, i.e., `StandardScaler()`.
    """
    scaler = MinMaxScaler()  # StandardScaler()
    scaler.fit(X_train)
    torch.save(scaler, args.data_save.format(args.exp_id, "train_scale"))
    # Scale features of X according to feature_range
    X_train = np.array(scaler.transform(X_train), dtype=np.float32)
    X_valid = np.array(scaler.transform(X_valid), dtype=np.float32)

# normalize the label.
# min_card_log = np.min([np.log(y) for y in y_train])  # -16.997074
# max_card_log = np.max([np.log(y) for y in y_train])  # 2.9952054
#
# normalize to (0, 1)
# y_train = [normalize(y, min_card_log, max_card_log) for y in y_train]
# y_valid = [normalize(y, min_card_log, max_card_log) for y in y_valid]
y_train = [np.log(y) for y in y_train]
y_valid = [np.log(y) for y in y_valid]

y_train, y_valid = np.array(y_train, dtype=np.float32), np.array(y_valid, dtype=np.float32)

# : 6. create the train/valid data loader.
train_set, valid_set = list(zip(X_train, y_train)), list(zip(X_valid, y_valid))

# torch.save(train_set, args.data_save.format(args.exp_id, "train"))
# torch.save(valid_set, args.data_save.format(args.exp_id, "valid"))
# logging.info(f"Save the training and testing dataset into {os.path.dirname(args.data_save.format(args.exp_id, 0))}.")

logging.info(f"Start the training process of the model of {args.model_type}.")

# https://xgboost.readthedocs.io/en/latest/parameter.html
if args.model_type == "XGBoost":
    X_train = [item[0] for item in list(train_set)]
    y_train = [item[1] for item in list(train_set)]
    X_valid = [item[0] for item in list(valid_set)]
    y_valid = [item[1] for item in list(valid_set)]

    # : adjust the params (classification and regression).
    # 333 train-rmse:136547.53125	test-rmse:411392.46875
    # 666 train-rmse:136547.54688	test-rmse:411392.50000
    # 66  train-rmse:136547.54688	test-rmse:411392.46875
    params = {"booster": "gbtree",  # gbtree, gblinear, dart
              "disable_default_eval_metric": False,
              "eta": 0.01,
              "max_depth": 15,  # larger <-> overfit
              "subsample": 0.7,
              "objective": "reg:linear",
              # "objective": "binary:logistic",
              # "eval_metric": "mape",
              "eval_metric": "rmse",
              "seed": args.seed}

    treeModel = XGBoost()
    treeModel.train(np.array(X_train), np.array(y_train), num_rounds=args.num_rounds, params=params)
    treeModel.save_model(args.model_save_dir.format(args.exp_id, "reg_xgb_cost"))

    y_pred = treeModel.estimate(X_valid)
    mse = mean_squared_error(y_valid, y_pred)
    rmse = mean_squared_error(y_valid, y_pred, squared=False)

    # 116.5750, 333.3838
    qerror = QError()(torch.tensor(np.exp(y_pred)), torch.tensor(np.exp(y_valid)))
    # qerror = QError()(torch.tensor(y_pred), torch.tensor(y_valid), out="raw")

    print("mse: ", mse, "rmse: ", rmse, "qerror: ", qerror)

# https://lightgbm.readthedocs.io/en/latest/Parameters.html
# https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
elif args.model_type == "LightGBM":
    X_train = [item[0] for item in list(train_set)]
    y_train = [item[1] for item in list(train_set)]
    X_valid = [item[0] for item in list(valid_set)]
    y_valid = [item[1] for item in list(valid_set)]

    # : adjust the params (classification and regression).
    params = {
        "task": "train",
        # "n_estimators": 100,
        "boosting_type": "gbdt",  # 设置提升类型
        "objective": "regression",  # 目标函数
        # "metric": {"mape"},  # {"l2", "l1"},  # 评估函数
        "metric": {"l2"},  # {"l2", "l1"},  # 评估函数
        # "metric": "None",
        "num_leaves": 80,  # 叶子节点数
        "learning_rate": 0.01,
        "feature_fraction": 0.9,  # 建树的特征选择比例
        "bagging_fraction": 0.8,  # 建树的样本采样比例
        "bagging_freq": 5,  # k 意味着每 k 次迭代执行bagging
        # "verbose": 1,  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
        "seed": args.seed
    }

    treeModel = LightGBM()
    treeModel.train(np.array(X_train), np.array(y_train), num_rounds=args.num_rounds, params=params)
    treeModel.save_model(args.model_save_dir.format(args.exp_id, "reg_lgb_cost"))

    y_pred = treeModel.estimate(X_valid)
    mse = mean_squared_error(y_valid, y_pred)
    # 76.7475, 284.5757
    qerror = QError()(torch.tensor(np.exp(y_pred)), torch.tensor(np.exp(y_valid)))
    print("mse: ", mse, "qerror: ", qerror.item())

elif args.model_type == "RandomForest":
    params = {"n_estimators": 40,  # 350
              "criterion": "squared_error",
              "random_state": 666,
              "max_depth": 10,  # 30
              "max_features": "sqrt",
              "min_samples_leaf": 1,
              "min_impurity_decrease": 1e-6,
              "verbose": 2}

    treeModel = RandomForest(task_type=args.task_type)

    grid_search = False
    if grid_search:
        param_grid = {
            "n_estimators": [250, 300, 350],
            "max_features": ["sqrt", "log2"],
            "max_depth": [15, 20, 25, 30]
        }
        gbm = GridSearchCV(treeModel.model, param_grid)
        gbm.fit(X_train, y_train)
        print('Best parameters found by grid search are:', gbm.best_params_)

    treeModel.train(X_train, y_train, params=params)
    treeModel.save_model(args.model_save_dir.format(args.exp_id, "reg_rf_cost"))

    y_pred = treeModel.estimate(X_valid)
    mse = mean_squared_error(y_valid, y_pred)
    qerror = QError()(torch.tensor(np.exp(y_pred)), torch.tensor(np.exp(y_valid)))
    print("mse: ", mse, "qerror: ", qerror.item())

logging.info(f"End the training process of the model of {args.model_type}.")
