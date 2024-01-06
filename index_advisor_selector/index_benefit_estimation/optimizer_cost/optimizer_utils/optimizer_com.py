# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: optimizer_com
# @Author: Wei Zhou
# @Time: 2023/10/6 21:58

import logging
import argparse

tf_step = 0
summary_writer = None


def get_parser():
    parser = argparse.ArgumentParser(
        description="A MODEL for cost refinement.")

    parser.add_argument("--exp_id", type=str, default="new_exp_opt")
    parser.add_argument("--gpu_no", type=str, default="0")

    parser.add_argument("--data_type", type=str, default="syntheti")
    parser.add_argument("--model_type", type=str, default="XGBoost",
                        choices=["XGBoost", "LightGBM", "RandomForest"])

    parser.add_argument("--plan_num", type=int, default=1)
    parser.add_argument("--feat_chan", type=str, default="cost_row",
                        choices=["cost", "row", "cost_row"])
    parser.add_argument("--label_type", type=str, default="raw",
                        choices=["ratio", "diff_ratio", "cla", "raw"])

    parser.add_argument("--feat_conn", type=str, default="concat")
    parser.add_argument("--task_type", type=str, default="reg")
    parser.add_argument("--cla_min_ratio", type=float, default=0.2)

    parser.add_argument("--data_load", type=str,
                        default="/data/wz/index/index_eab/eab_benefit/tree_model/data/job_cost_data_src_tree.json")
    parser.add_argument("--train_data_load", type=str,
                        default="/data/wz/index/index_eab/eab_benefit/tree_model/data/job_cost_data_src_tree.json")
    parser.add_argument("--valid_data_load", type=str,
                        default="/data/wz/index/index_eab/eab_benefit/tree_model/data/job_cost_data_src_tree.json")

    parser.add_argument("--data_save", type=str,
                        default="./cost_exp_res/{}/data/{}_data.pt")

    parser.add_argument("--model_save_gap", type=int, default=1)
    parser.add_argument("--model_save", type=str,
                        default="./cost_exp_res/{}/model/cost_{}.pt")
    parser.add_argument("--model_save_dir", type=str,
                        default="./cost_exp_res/{}/model/{}")

    # : 1. common setting.
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--logdir", type=str,
                        default="./cost_exp_res/{}/logdir/")
    parser.add_argument("--runlog", type=str,
                        default="./cost_exp_res/{}/exp_runtime.log")

    # : hyper parameter.
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--num_round", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hid_dim", type=int, default=128)

    return parser


def set_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # log to file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)


def add_summary_value(key, value, step=None):
    if step is None:
        summary_writer.add_scalar(key, value, tf_step)
    else:
        summary_writer.add_scalar(key, value, step)
