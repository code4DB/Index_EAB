import os
import random
import sys
import copy
import json
import logging
import pickle
import argparse
import configparser

from index_advisor_selector.index_selection.heu_selection.heu_utils.workload import Workload, Table, Column, Query
from index_advisor_selector.index_selection.heu_selection.heu_utils.constants import tpch_tables, tpcds_tables, job_table_alias


def get_parser():
    parser = argparse.ArgumentParser(
        description="the ISP solved by Heuristic-based Methods.")

    parser.add_argument("--algo", type=str, default=None)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--sel_params", type=str, default="parameters")
    parser.add_argument("--exp_conf_file", type=str,
                        default="/data/wz/index/index_eab/eab_data/heu_run_conf/{}_config_tpch.json")

    parser.add_argument("--constraint", type=str, default="storage",
                        choices=["storage", "number"])
    parser.add_argument("--budget_MB", type=int, default=500)
    parser.add_argument("--max_indexes", type=int, default=5)

    parser.add_argument("--max_index_width", type=int, default=None)

    parser.add_argument("--process", action="store_true")
    parser.add_argument("--overhead", action="store_true")
    parser.add_argument("--varying_frequencies", action="store_true")

    parser.add_argument("--cand_gen", type=str, default=None,
                        choices=["permutation", "dqn_rule", "openGauss"])
    parser.add_argument("--is_utilized", action="store_true")  # , default=True
    parser.add_argument("--multi_column", action="store_true")  # , default=True

    parser.add_argument("--est_model", type=str, default="optimizer",
                        choices=["optimizer", "tree", "lib", "queryformer"])

    parser.add_argument("--sel_oracle", type=str, default=None,
                        choices=["cost_per_sto", "cost_pure", "benefit_per_sto", "benefit_pure"])

    parser.add_argument("--work_file", type=str,
                        default="/data/wz/index/index_eab/eab_olap/bench_temp/tpch_template_18.sql")

    parser.add_argument("--db_conf_file", type=str,
                        default="/data/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpch_1gb.conf")
    parser.add_argument("--schema_file", type=str,
                        default="/data/wz/index/index_eab/eab_data/db_info_conf/schema_tpch_1gb.json")

    parser.add_argument("--res_save", type=str)  # , required=True

    # (1211): newly added. for `cophy`
    parser.add_argument("--ampl_solver", type=str, default="highs")
    parser.add_argument("--ampl_bin_path", type=str,
                        default="/data1/wz/ampl.linux-intel64")
    parser.add_argument("--ampl_mod_path", type=str,
                        default="/data1/wz/index/index_eab/eab_data/heu_run_conf/cophy_ampl_model_{}.mod")
    parser.add_argument("--ampl_dat_path", type=str,
                        default="/data1/wz/index/index_eab/eab_olap/bench_result/tpch/"
                                "work_level/tpch_1gb_template_18_multi_work_index_cophy.txt")

    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--db_name", type=str, default=None)
    parser.add_argument("--port", type=str, default=None)
    parser.add_argument("--user", type=str, default=None)
    parser.add_argument("--password", type=str, default=None)

    return parser


def get_conf(conf_file):
    conf = configparser.ConfigParser()
    conf.read(conf_file)

    return conf


def parse_command_line_args():
    arguments = sys.argv
    if "CRITICAL_LOG" in arguments:
        logging.getLogger().setLevel(logging.CRITICAL)
    if "ERROR_LOG" in arguments:
        logging.getLogger().setLevel(logging.ERROR)
    if "INFO_LOG" in arguments:
        logging.getLogger().setLevel(logging.INFO)
    for argument in arguments:
        if ".json" in argument:
            return argument


def set_logger(log_file):
    # logging.basicConfig(
    #     filename=log_file,
    #     filemode='w',
    #     format='%(asctime)s - %(levelname)s - %(message)s')

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


def find_parameter_list(algorithm_config, params="parameters"):
    """
    # Parameter list example: {"max_indexes": [5, 10, 20]}
    # Creates config for each value.

    :param algorithm_config:
    :param params:
    :return:
    """
    parameters = algorithm_config[params]
    configs = list()
    if parameters:
        # if more than one list --> raise
        # Only support one param list in each algorithm.
        counter = 0
        for key, value in parameters.items():
            if isinstance(value, list):
                counter += 1
        if counter > 1:
            raise Exception("Too many parameter lists in config.")

        for key, value in parameters.items():
            if isinstance(value, list):
                for i in value:
                    new_config = copy.deepcopy(algorithm_config)
                    new_config["parameters"][key] = i
                    configs.append(new_config)
    if len(configs) == 0:
        configs.append(algorithm_config)

    return configs


def get_columns_from_db(db_connector):
    # db_connector.create_connection()

    tables, columns = list(), list()
    for table in db_connector.get_tables():
        table_object = Table(table)
        tables.append(table_object)
        for col in db_connector.get_cols(table):
            column_object = Column(col)
            table_object.add_column(column_object)
            columns.append(column_object)

    # db_connector.close()

    return tables, columns


def get_columns_from_schema(schema_file):
    tables, columns = list(), list()
    with open(schema_file, "r") as rf:
        db_schema = json.load(rf)

    for item in db_schema:
        table_object = Table(item["table"])
        tables.append(table_object)
        for col_info in item["columns"]:
            column_object = Column(col_info["name"])
            table_object.add_column(column_object)
            columns.append(column_object)

    return tables, columns


def read_row_query(sql_list, exp_conf, columns, type="template",
                   varying_frequencies=False, seed=666):
    random.seed(seed)

    workload = list()
    for query_id, query_text in enumerate(sql_list):
        if type == "template" and exp_conf["queries"] \
                and query_id + 1 not in exp_conf["queries"]:
            continue

        # (0824): newly modified.
        if isinstance(query_text, list):
            if varying_frequencies:
                freq = query_text[-1]
            else:
                freq = 1
            query = Query(query_text[0], query_text[1], frequency=freq)
        elif isinstance(query_text, str):
            if varying_frequencies:
                freq = random.randint(1, 1000)
            else:
                freq = 1
            query = Query(query_id, query_text, frequency=freq)

        for column in columns:
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
                    # if column.name in query.text and column.table.name in query.text:
                    # if " " + column.name + " " in query.text and column.table.name in query.text:
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
                    # else:
                    #     # (0408): newly added. check?
                    #     # if column.name in query.text:
                    #     if column.name in query.text.lower() and \
                    #             f"{column.table.name}" in query.text.lower():
                    #         query.columns.append(column)

                # (0408): newly added. check? (different table, same column name)
                # if column.name in query.text.lower() and \
                #         column.table.name in query.text.lower():
                #     query.columns.append(column)
                # if column.name in query.text:
                #     query.columns.append(column)
        workload.append(query)

    logging.info("Queries read.")
    return workload


def read_row_query_new(sql_list, columns):
    workload = list()
    for query_id, query_text in enumerate(sql_list):
        query = Query(query_id, query_text)
        for column in columns:
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
                    # if column.name in query.text and column.table.name in query.text:
                    # if " " + column.name + " " in query.text and column.table.name in query.text:
                    # (0329): newly modified. for JOB,
                    #  SELECT COUNT(*), too many candidates.
                    if "." in query.text.lower().split("from")[0] or \
                            ("where" in query.text.lower() and (
                                    "." in query.text.lower().split("where")[0] or
                                    "." in query.text.lower().split("where")[-1].split(" ")[1])):
                        if str(column) in query.text.lower():
                            query.columns.append(column)
                        if " as " in query_text.lower():
                            tbl, col = str(column).split(".")
                            if f" {job_table_alias[tbl]}.{col}" in query.text.lower() \
                                    or f"({job_table_alias[tbl]}.{col}" in query.text.lower():
                                query.columns.append(column)
                    # else:
                    #     # (0408): newly added. check?
                    #     # if column.name in query.text:
                    #     if column.name in query.text.lower() and \
                    #             f"{column.table.name}" in query.text.lower():
                    #         query.columns.append(column)

                # (0408): newly added. check? (different table, same column name)
                # if column.name in query.text.lower() and \
                #         column.table.name in query.text.lower():
                #     query.columns.append(column)
                # if column.name in query.text:
                #     query.columns.append(column)
        workload.append(query)

    logging.info("Queries read.")
    return workload


# --- Unit conversions ---
# Storage
def b_to_mb(b):
    """
    1024?
    :param b:
    :return:
    """
    return b / 1000 / 1000


def mb_to_b(mb):
    return mb * 1000 * 1000


# Time
def s_to_ms(s):
    return s * 1000


# --- Index selection utilities ---
def indexes_by_table(indexes):
    indexes_by_table = dict()
    # (0804): newly added. for reproduction.
    for index in sorted(list(indexes)):
        table = index.table()
        if table not in indexes_by_table:
            indexes_by_table[table] = list()
        indexes_by_table[table].append(index)

    return indexes_by_table


def get_utilized_indexes(
        workload, indexes_per_query, cost_evaluation, detailed_query_information=False
):
    utilized_indexes_workload = set()
    query_details = {}
    for query, indexes in zip(workload.queries, indexes_per_query):
        (
            utilized_indexes_query,
            cost_with_indexes,
        ) = cost_evaluation.which_indexes_utilized_and_cost(query, indexes)
        utilized_indexes_workload |= utilized_indexes_query

        if detailed_query_information:
            cost_without_indexes = cost_evaluation.calculate_cost(
                Workload([query]), indexes=[]
            )
            # : cost_with_indexes > cost_without_indexes, continue.
            query_details[query] = {
                "cost_without_indexes": cost_without_indexes,
                "cost_with_indexes": cost_with_indexes,
                "utilized_indexes": utilized_indexes_query,
            }

    return utilized_indexes_workload, query_details
