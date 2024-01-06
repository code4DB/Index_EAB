# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: mcts_com
# @Author: Wei Zhou
# @Time: 2022/11/1 21:07

import json
import random
import logging
import itertools

import argparse
import seaborn as sns

from .mcts_workload import Table, Column, Query, Index
from .mcts_const import tpch_tables, tpcds_tables, job_table_alias


def get_parser():
    parser = argparse.ArgumentParser(
        description="the MCTS Version of Index Selection Algorithm.")

    parser.add_argument("--exp_id", type=str, default="mcts_new_exp")
    parser.add_argument("--is_trace", action="store_true")
    parser.add_argument("--process", action="store_true")  # , default=True
    parser.add_argument("--overhead", action="store_true")
    parser.add_argument("--varying_frequencies", action="store_true")

    # (1118): newly added.
    parser.add_argument("--is_utilized", action="store_true")
    parser.add_argument("--cand_gen", default=None)

    # 1. MCTS configuration.
    parser.add_argument("--mcts_seed", type=int, default=666)
    parser.add_argument("--constraint", type=str, default="storage",
                        choices=["number", "storage"],
                        help="The tuning constraint of the MCTS selection.")

    parser.add_argument("--budget", type=int, default=50,
                        help="The number of the MCTS iterations.")
    parser.add_argument("--cardinality", type=int, default=5,
                        help="The number of the maximum indexes.")
    parser.add_argument("--storage", type=int, default=500,
                        help="The value of the maximum storage budget.")

    parser.add_argument("--min_budget", type=int, default=100)
    parser.add_argument("--early_stopping", type=int, default=5)

    parser.add_argument("--max_index_width", type=int, default=2)
    parser.add_argument("--select_policy", type=str, default="UCT",
                        choices=["UCT", "EPSILON"])
    parser.add_argument("--roll_num", type=int, default=1,
                        help="The number of the roll out.")
    parser.add_argument("--best_policy", type=str, default="BG",
                        choices=["BCE", "BG"])

    # 2. Common configuration.
    parser.add_argument("--work_file", type=str,
                        default="/data/wz/index/index_eab/eab_olap/bench_temp/tpch/tpch_work_temp_multi_w18_n10_eval.json")
    parser.add_argument("--schema_file", type=str,
                        default="/data/wz/index/index_eab/eab_data/db_info_conf/schema_tpch_1gb.json")
    parser.add_argument("--db_file", type=str,
                        default="/data/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpch_1gb.conf")
    parser.add_argument("--res_save", type=str,
                        default="/data/wz/index/index_eab/eab_olap/bench_temp/tpch/tpch_work_temp_multi_w18_n10_eval.json")

    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--db_name", type=str, default=None)
    parser.add_argument("--port", type=str, default=None)
    parser.add_argument("--user", type=str, default=None)
    parser.add_argument("--password", type=str, default=None)

    parser.add_argument("--log_file", type=str,
                        default="./exp_res/{}/mcts_exp_runtime.log")

    parser.add_argument("--model_load", type=str,
                        default="/data/wz/index/index_eab/eab_algo/mcts_selection/exp_res/mcts_exp/mcts_tree.pickle")
    parser.add_argument("--model_save", type=str,
                        default="./exp_res/{}/mcts_tree.pickle")

    return parser


def set_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
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


def plot_report(save_dir, measure, log_y=False, save_path=None,
                save_conf={"format": "pdf", "bbox_inches": "tight"}):
    x_label = "Epoch"
    for name, values in measure.items():
        sns_plot = sns.relplot(kind="line", errorbar="sd", data=values, err_style="band")

        if log_y:
            sns_plot.set(yscale="log")

        sns_plot.set(xlabel=x_label, ylabel=name)

        save_path = f"{save_dir}/{name}.png"
        sns_plot.savefig(save_path)


def pre_work(work_list, schema_file, varying_frequencies=False):
    # 1. prepare the database schema.
    with open(schema_file, "r") as rf:
        schema_json = json.load(rf)

    tables = list()
    for tbl_info in schema_json:
        tbl = Table(tbl_info["table"])
        for col_info in tbl_info["columns"]:
            tbl.add_column(Column(col_info["name"]))
        tables.append(tbl)

    columns = list()
    for tbl in tables:
        columns.extend(tbl.columns)

    # 2. load the workload.
    workload = list()
    for i, sql_text in enumerate(work_list):
        if isinstance(sql_text, list):
            if varying_frequencies:
                freq = sql_text[-1]
            else:
                freq = 1
            query = Query(sql_text[0], sql_text[1], frequency=freq)
        elif isinstance(sql_text, str):
            if varying_frequencies:
                freq = random.randint(1, 1000)
            else:
                freq = 1
            query = Query(i, sql_text, frequency=freq)

        for column in columns:
            # if column.name in query:
            # # if column.name in query and column.table.name in query:
            # # if " " + column.name + " " in query and column.table.name in query:
            #     columns.append(col)
            # (0329): newly modified. for JOB,
            #  SELECT COUNT(*), too many candidates.
            # if "." in query.lower().split("from")[0] or \
            #         ("where" in query.lower() and ("." in query.lower().split("where")[0] or
            #                                        "." in query.lower().split("where")[-1].split(" ")[
            #                                            1])):
            #     if str(column) in query.lower():
            #         columns.append(column)
            # else:
            #     # (0408): newly added. check?
            #     if column.name in query.lower() and \
            #             f"{column.table.name}" in query.lower():
            #         columns.append(column)

            column_tmp = [col for col in columns if column.name == col.name]
            if len(column_tmp) == 1:
                if column.name in query.text.lower() and \
                        f"{column.table.name}" in query.text.lower():
                    query.columns.append(column)
            else:
                if column.table.name not in tpch_tables + tpcds_tables + list(job_table_alias.keys()):
                    if column.name in query.text.lower() and \
                            f"{column.table.name}" in query.text.lower():
                        query.columns.append(column)
                else:
                    # (0329): newly modified. for JOB,
                    #  SELECT COUNT(*), too many candidates.
                    if "." in query.text.lower().split("from")[0] or \
                            ("where" in query.text.lower() and (
                                    "." in query.text.lower().split("where")[0] or
                                    "." in query.text.lower().split("where")[-1].split(" ")[1])):
                        if str(column) in query.text.lower():
                            query.columns.append(column)
                        if " as " in query.text.lower():
                            tbl, col = str(column).split(".")
                            if f" {job_table_alias[tbl]}.{col}" in query.text.lower() \
                                    or f"({job_table_alias[tbl]}.{col}" in query.text.lower():
                                query.columns.append(column)

        workload.append(query)

    return workload


def syntactically_relevant_indexes(workload, max_index_width):
    # "SAEFIS" or "BFI" see paper linked in DB2Advis algorithm
    # This implementation is "BFI" and uses all syntactically relevant indexes.
    possible_column_combinations = set()

    # (1118): newly added.
    if not isinstance(workload, list):
        workload = [workload]

    for query in workload:
        columns = query.columns
        logging.debug(f"{query}")
        logging.debug(f"Indexable columns: {len(columns)}")

        indexable_columns_per_table = {}
        for column in columns:
            if column.table not in indexable_columns_per_table:
                indexable_columns_per_table[column.table] = set()
            indexable_columns_per_table[column.table].add(column)

        for table in indexable_columns_per_table:
            columns = indexable_columns_per_table[table]
            for index_length in range(1, max_index_width + 1):
                possible_column_combinations |= set(
                    itertools.permutations(columns, index_length)
                )
        logging.debug(f"Potential indexes: {len(possible_column_combinations)}")

    # indexes = [",".join(map(str, ind)) for ind in possible_column_combinations]
    # cols = [ind.split(",") for ind in indexes]
    # cols = [list(map(lambda x: x.split(".")[-1], col)) for col in cols]
    # indexes = [f"{ind.split('.')[0]}#{','.join(col)}"
    #            for ind, col in zip(indexes, cols)]

    # return sorted(indexes)
    return sorted([Index(p) for p in possible_column_combinations])


def mb_to_b(mb):
    return mb * 1000 * 1000
