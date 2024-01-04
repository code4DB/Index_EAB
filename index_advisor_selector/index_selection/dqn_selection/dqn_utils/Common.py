# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: Common
# @Author: Wei Zhou
# @Time: 2023/8/8 10:10

import logging
import argparse
import psqlparse

import matplotlib.pyplot as plt

tf_step = 0


def get_parser():
    parser = argparse.ArgumentParser(
        description="The ISP solved by DQN.")

    parser.add_argument("--gpu_no", type=str, default="-1")
    parser.add_argument("--exp_id", type=str, default="new_exp")
    parser.add_argument("--epoch", type=int, default=10)

    parser.add_argument("--action_mode", type=str, default="train",
                        choices=["train", "infer"])

    parser.add_argument("--constraint", type=str, default="storage",
                        choices=["number", "storage"])
    parser.add_argument("--max_count", type=int, default=5)
    parser.add_argument("--max_storage", type=int, default=500)

    parser.add_argument("--pre_create", action="store_true")

    parser.add_argument("--is_dnn", action="store_true")
    parser.add_argument("--is_ps", action="store_true")
    parser.add_argument("--is_double", action="store_true")
    parser.add_argument("--a", type=float, default=0,
                        help="Control the items in the calculation of a reward")

    parser.add_argument("--conf_load", type=str,
                        default="/data/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpch_1gb.conf")
    parser.add_argument("--work_load", type=str,
                        default="/data/wz/index/index_eab/eab_olap/bench_temp/tpch_template_18.sql")
    parser.add_argument("--cand_load", type=str,
                        default="empty")
    parser.add_argument("--model_load", type=str,
                        default="empty")

    parser.add_argument("--runlog", type=str,
                        default="./exp_res/{}/exp_runtime.log")
    parser.add_argument("--logdir", type=str,
                        default="./exp_res/{}/logdir/")
    parser.add_argument("--model_save", type=str,
                        default="./exp_res/{}/model/dqn_{}.pt")
    parser.add_argument("--save_gap", type=int, default=20)

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


def gen_cands(workload, sql_parser):
    for no, query in enumerate(workload):
        b = psqlparse.parse_dict(query)
        sql_parser.parse_stmt(b[0])
        sql_parser.gain_candidates()

    cands = sql_parser.index_candidates
    cands = list(cands)
    cands.sort()

    return cands


def plot_report(exp_dir, measure, save_path=None,
                save_conf={"format": "pdf", "bbox_inches": "tight"}):
    no = 0
    x_label = "Epoch"
    for name, values in measure.items():
        # sns_plot = sns.relplot(kind="line", errorbar="sd", data=values, err_style="band")
        # if log_y:
        #     sns_plot.set(yscale="log")
        # sns_plot.set(xlabel=x_label, ylabel=name)

        plt.figure(no)
        x = range(len(values))
        y = values
        plt.plot(x, y)
        # plt.plot(x, y, marker="x")
        plt.xlabel(x_label)
        plt.ylabel(name)

        save_path = f"{exp_dir}/{name}.png"
        plt.savefig(save_path, dpi=120)

        no += 1
