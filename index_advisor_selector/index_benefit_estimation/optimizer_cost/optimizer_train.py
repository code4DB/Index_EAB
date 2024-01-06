# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: optimizer_train
# @Author: Wei Zhou
# @Time: 2023/10/6 19:54

import os
import json
import logging

import random
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error

from index_advisor_selector.index_benefit_estimation.optimizer_cost.optimizer_utils import optimizer_com
from index_advisor_selector.index_benefit_estimation.optimizer_cost.optimizer_utils.optimizer_loss import cal_mape, QError
from index_advisor_selector.index_benefit_estimation.optimizer_cost.optimizer_model import Optimizer

# : 1. get the params.
parser = optimizer_com.get_parser()
args = parser.parse_args()

import torch
from torch.utils.tensorboard import SummaryWriter

# : 3. create the directory to store the `exp_res`.
# assert not os.path.exists(os.path.dirname(args.logdir.format(args.exp_id))), \
#     f"`{os.path.dirname(args.logdir.format(args.exp_id))}` dir existed!"
os.makedirs(os.path.dirname(args.logdir.format(args.exp_id)))
os.makedirs(os.path.dirname(args.model_save.format(args.exp_id, 0)))
os.makedirs(os.path.dirname(args.data_save.format(args.exp_id, 0)))

optimizer_com.set_logger(args.runlog.format(args.exp_id))
logging.info("Start Cost Calibration Experiment.")

logging.info(f"Create the directory `{os.path.dirname(args.logdir.format(args.exp_id))}` to save experiment result.")

# specify the path to store the exp_res of `logdir` of the tensorboard.
optimizer_com.summary_writer = SummaryWriter(args.logdir.format(args.exp_id))
optimizer_com.summary_writer.add_text(
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
args.train_data_load = "/data/wz/index/index_eab/eab_benefit/cost_data/tpch/tpch_cost_data_src_train.json"
# args.train_data_load = "/data/wz/index/index_eab/eab_benefit/optimizer_cost/data/tpch/openGauss_tpch_cost_data_src_train.json"
with open(args.train_data_load, "r") as rf:
    train_data = json.load(rf)

args.valid_data_load = "/data/wz/index/index_eab/eab_benefit/cost_data/tpch/tpch_cost_data_src_valid.json"
# args.valid_data_load = "/data/wz/index/index_eab/eab_benefit/optimizer_cost/data/tpch/openGauss_tpch_cost_data_src_valid.json"
with open(args.valid_data_load, "r") as rf:
    valid_data = json.load(rf)

logging.info(f"Load the train data from `{args.train_data_load}` ({len(train_data)}).")
logging.info(f"Load the valid data from `{args.valid_data_load}` ({len(valid_data)}).")

x_train = np.log(np.array([item["w/ estimated cost"] for item in train_data])).reshape(-1, 1)
y_train = np.log(np.array([item["w/ actual cost"] for item in train_data])).reshape(-1, 1)

opt = Optimizer()
opt.train(x_train, y_train)

x_valid = np.log(np.array([item["w/ estimated cost"] for item in valid_data])).reshape(-1, 1)
y_valid = np.log(np.array([item["w/ actual cost"] for item in valid_data])).reshape(-1, 1)

y_pred = opt.estimate(x_valid)

# 13713.305151052697, 13269.00674627733
mse = mean_squared_error(y_pred, y_valid, squared=False)
# 24921.1201, 26566.0884
qerror = QError()(torch.tensor(np.exp(y_pred)), torch.tensor(np.exp(y_valid)), out="raw")
# qerror = QError()(torch.tensor(y_pred), torch.tensor(y_valid))
qerror = qerror.numpy()
print("mse: ", mse, "mean qerror: ", np.mean(qerror), "median qerror: ", np.median(qerror),
      "90th qerror", np.quantile(qerror, 0.9), "95th qerror", np.quantile(qerror, 0.95))

logging.info(f"End the training process of the model of {args.model_type}.")
