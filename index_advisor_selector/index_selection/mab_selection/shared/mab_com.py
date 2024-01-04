# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: mab_com
# @Author: Wei Zhou
# @Time: 2023/8/15 21:01

import logging
import argparse

import seaborn as sns

from shared.helper import get_experiment_folder_path

tf_step = 0


def get_parser():
    parser = argparse.ArgumentParser(
        description="The ISP solved by MAB.")

    parser.add_argument("--exp_id", type=str, default="mab_new_exp")
    parser.add_argument("--bench", type=str, default="tpch",
                        choices=["tpch", "tpch_skew", "tpcds", "dsb", "job"])
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--min_rounds", type=int, default=50)

    parser.add_argument("--process", action="store_true")
    parser.add_argument("--varying_frequencies", action="store_true")

    parser.add_argument("--constraint", type=str, default="storage",
                        choices=["number", "storage"])
    parser.add_argument("--max_count", type=int, default=5)
    parser.add_argument("--max_memory", type=int, default=500)
    parser.add_argument("--early_stopping", type=int, default=10)

    parser.add_argument("--exp_file", type=str,
                        default="/data/wz/index/index_eab/eab_algo/mab_selection/config/exp.conf")
    parser.add_argument("--db_file", type=str,
                        default="/data/wz/index/index_eab/eab_algo/mab_selection/config/db.conf")
    parser.add_argument("--workload_file", type=str,
                        default="/data/wz/index/index_eab/eab_olap/bench_temp/tpch_template_18.sql")
    parser.add_argument("--schema_file", type=str,
                        default="/data/wz/index/index_eab/eab_data/db_info_conf/schema_tpch_1gb.json")
    parser.add_argument("--res_save", type=str,
                        default="/data/wz/index/index_eab/eab_olap/bench_temp/tpch/tpch_work_temp_multi_w18_n10_eval.json")

    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--db_name", type=str, default=None)
    parser.add_argument("--port", type=str, default=None)
    parser.add_argument("--user", type=str, default=None)
    parser.add_argument("--password", type=str, default=None)

    parser.add_argument("--params_load", type=str,
                        default="empty")
    parser.add_argument("--params_save", type=str,
                        default="./experiments/{}/model/mab_{}.pt")
    parser.add_argument("--save_gap", type=int, default=20)

    return parser


def set_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # log to file
    fh = logging.FileHandler(log_file, mode="w")  # mode="w"
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)


def plot_report(exp_id, measure, log_y=False, save_path=None,
                save_conf={"format": "pdf", "bbox_inches": "tight"}):
    x_label = "Epoch"
    for name, values in measure.items():
        # sns_plot = sns.relplot(x=x_label, y=name, hue="hue",
        #                        kind="line", errorbar="sd", data=values, err_style="band")
        sns_plot = sns.relplot(kind="line", errorbar="sd", data=values, err_style="band")

        if log_y:
            sns_plot.set(yscale="log")

        sns_plot.set(xlabel=x_label, ylabel=name)

        save_path = f"{get_experiment_folder_path(exp_id)}/{name}.png"
        sns_plot.savefig(save_path)
