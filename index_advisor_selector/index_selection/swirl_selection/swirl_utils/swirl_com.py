import json
import pickle
import logging

import copy
import argparse
import itertools

import psqlparse

from .index import Index
from .workload import Workload, Query, Table, Column
from .postgres_dbms import PostgresDatabaseConnector
from .constants import tpch_tables, tpcds_tables, job_table_alias

from .cost_evaluation import CostEvaluation

import index_advisor_selector.index_selection.dqn_selection.dqn_utils.Encoding as en
import index_advisor_selector.index_selection.dqn_selection.dqn_utils.ParserForIndex as pi

excluded_qno = {"tpch": [20 - 1, 17 - 1, 18 - 1],
                "tpcds": [2, 29, 36, 56, 87, 89, 95,
                          3, 34, 55, 73,
                          21, 25, 16,
                          6, 39],
                "job": []}


def get_parser():
    parser = argparse.ArgumentParser(
        description="the ISP solved by RL-based Methods.")

    parser.add_argument("--exp_id", type=str, default="swirl_new_exp")
    parser.add_argument("--algo", type=str, default="swirl",
                        choices=["swirl", "drlinda", "dqn"])
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--constraint", type=str, default="storage",
                        choices=["storage", "number"])
    parser.add_argument("--max_budgets", type=int, default=500)
    parser.add_argument("--max_indexes", type=int, default=5)

    parser.add_argument("--is_query_cache", action="store_true")
    parser.add_argument("--training_instances", type=int, default=None)
    parser.add_argument("--validation_testing_instances", type=int, default=None)
    parser.add_argument("--varying_frequencies", action="store_true")

    parser.add_argument("--exp_conf_file", type=str,
                        default="/data/wz/index/index_eab/eab_data/rl_run_conf/swirl_tpch_1gb.json")
    parser.add_argument("--db_conf_file", type=str,
                        default="/data/wz/index/index_eab/eab_data/db_info_conf/local_db103_tpch_1gb.conf")
    parser.add_argument("--schema_file", type=str,
                        default="/data/wz/index/index_eab/eab_data/db_info_conf/schema_tpch_1gb.json")
    parser.add_argument("--colinfo_load", type=str,
                        default="/data/wz/index/index_eab/eab_data/db_info_conf/colinfo_tpch_1gb.json")

    parser.add_argument("--user", type=str, default=None)
    parser.add_argument("--password", type=str, default=None)
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--db_name", type=str, default=None)
    parser.add_argument("--port", type=str, default=None)

    parser.add_argument("--work_num", type=int, default=-1)
    # parser.add_argument("--temp_num", type=int, default=None)
    parser.add_argument("--temp_num", type=int, default=22)
    parser.add_argument("--class_num", type=int, default=-1)

    # parser.add_argument("--work_size", type=int, default=18)
    parser.add_argument("--work_size", type=int, default=1)
    parser.add_argument("--work_gen", type=str, default="load",
                        choices=["random", "load"])
    parser.add_argument("--filter_workload_cols", action="store_true")
    parser.add_argument("--filter_utilized_columns", action="store_true")

    # (0918): newly added.
    parser.add_argument("--max_index_width", type=int, default=None)
    parser.add_argument("--cand_gen", type=str, default=None,
                        choices=["permutation", "dqn_rule", "openGauss"])
    parser.add_argument("--action_manager", type=str, default=None)

    parser.add_argument("--workload_embedder", type=str, default=None,
                        choices=["PlanEmbedderPCA", "PlanEmbedderLSIBOW", "PlanEmbedderDoc2Vec",
                                 "SQLWorkloadPCA", "SQLWorkloadLSI", "SQLWorkloadDoc2Vec"])
    parser.add_argument("--observation_manager", type=str, default=None)

    parser.add_argument("--SCALER", type=int, default=None)
    parser.add_argument("--reward_calculator", type=str, default=None)

    # 1) template
    parser.add_argument("--eval_file", type=str,
                        default="/data/wz/index/index_eab/eab_olap/bench_temp/tpch/tpch_work_temp_multi_w18_freq_n10_eval.json")

    # 2) not_template
    parser.add_argument("--work_type", type=str, default="not_template",
                        choices=["template", "not_template"])
    parser.add_argument("--work_file", type=str,
                        default="/data/wz/index/index_eab/eab_olap/bench_temp/tpch/tpch_query_temp_multi_n1000.json")

    parser.add_argument("--temp_expand", action="store_true")
    parser.add_argument("--temp_load", type=str, default=None)
    
    parser.add_argument("--rl_exp_load", type=str,
                        default="/data/wz/index/attack/swirl_selection/exp_res/s152_swirlh1gb_temp_w18_b500_10w/experiment_object.pickle")
    parser.add_argument("--rl_model_load", type=str,
                        default="/data/wz/index/attack/swirl_selection/exp_res/s152_swirlh1gb_temp_w18_b500_10w/best_mean_reward_model.zip")
    parser.add_argument("--rl_env_load", type=str,
                        default="/data/wz/index/attack/swirl_selection/exp_res/s152_swirlh1gb_temp_w18_b500_10w/vec_normalize.pkl")

    parser.add_argument("--res_save_path", type=str, default="./exp_res",
                        help="The experimental result's folder.")
    parser.add_argument("--res_save", type=str, default=None,
                        help="The inference result's folder.")
    parser.add_argument("--logdir", type=str,
                        default="./exp_res/{}/logdir")
    parser.add_argument("--log_file", type=str,
                        default="./exp_res/{}/exp_runtime.log")

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


# : This could be improved by passing index candidates as input.
def predict_index_sizes(column_combinations, db_config, is_precond=True):
    connector = PostgresDatabaseConnector(db_config, autocommit=True)
    connector.drop_indexes()

    cost_evaluation = CostEvaluation(connector)

    predicted_index_sizes = []
    parent_index_size_map = {}
    for column_combination in column_combinations:
        potential_index = Index(column_combination)
        cost_evaluation.what_if.simulate_index(potential_index, store_size=True)

        full_index_size = potential_index.estimated_size
        index_delta_size = full_index_size

        # (1212): leading index? index_delta_size
        if is_precond:
            if len(column_combination) > 1:
                index_delta_size -= parent_index_size_map[column_combination[:-1]]

        predicted_index_sizes.append(index_delta_size)
        cost_evaluation.what_if.drop_simulated_index(potential_index)

        parent_index_size_map[column_combination] = full_index_size

    # : newly added.
    connector.close()

    return predicted_index_sizes


def get_hypo_index_sizes(column_combinations, db_config):
    connector = PostgresDatabaseConnector(db_config, autocommit=True)
    connector.drop_indexes()

    cost_evaluation = CostEvaluation(connector)

    predicted_index_sizes = []
    parent_index_size_map = {}
    for column_combination in column_combinations:
        potential_index = Index(column_combination)
        cost_evaluation.what_if.simulate_index(potential_index, store_size=True)

        full_index_size = potential_index.estimated_size
        index_delta_size = full_index_size
        if len(column_combination) > 1 and column_combination[:-1] in parent_index_size_map.keys():
            index_delta_size -= parent_index_size_map[column_combination[:-1]]

        predicted_index_sizes.append(index_delta_size)
        cost_evaluation.what_if.drop_simulated_index(potential_index)

        parent_index_size_map[column_combination] = full_index_size

    # : newly added.
    connector.close()

    return predicted_index_sizes


def get_prom_index_candidates(token_load, colinfo_load, columns):
    """
    1) `Join`: table join column;
    2) `Filter`: equal/range column;
    3) `Aggregate`: group by column;
    4) `Sort`: order by column.
    :return:
    """
    result_column_combinations = list()

    with open(token_load, "r") as rf:
        sql_tokens = json.load(rf)
    with open(colinfo_load, "r") as rf:
        col_info = json.load(rf)

    column_dict = dict()  # w_warehouse_sk
    for col in columns:
        #  (0327, 0418): newly added. for job and real.
        if "tpch" in colinfo_load or "tpcds" in colinfo_load:
            column_dict[str(col).split(".")[-1]] = col
        else:
            column_dict[str(col)] = col

    table_column_dict = dict()
    for column in columns:
        if column.table not in table_column_dict:
            table_column_dict[column.table] = set()
        table_column_dict[column.table].add(column)

    prom_ind = {1: list()}
    # all the single-column index.
    for col in column_dict.values():
        prom_ind[1].append(tuple([col]))
    for sql_token in sql_tokens:
        # 1. extract the columns in certain positions.
        join_col = list()
        for typ, tok in zip(sql_token["from"]["pre_type"], sql_token["from"]["pre_token"]):
            #  (0327, 0418): newly added. for job and real.
            if "tpch" in colinfo_load or "tpcds" in colinfo_load:
                if typ == "from#join_column" and tok.split(".")[-1] not in join_col \
                        and tok.split(".")[-1] in column_dict.keys():
                    join_col.append(tok.split(".")[-1])
            else:
                if typ == "from#join_column" and tok not in join_col \
                        and tok in column_dict.keys():
                    join_col.append(tok)

        eq_col, range_col = list(), list()
        if "where" in sql_token.keys():
            for i, tok in enumerate(sql_token["where"]["pre_token"]):
                #  (0327, 0418): newly added. for job and real.
                if "tpch" in colinfo_load or "tpcds" in colinfo_load:
                    if tok == "=" and \
                            sql_token["where"]["pre_token"][i - 1].split(".")[-1] not in eq_col \
                            and sql_token["where"]["pre_token"][i - 1].split(".")[-1] in column_dict.keys():
                        eq_col.append(sql_token["where"]["pre_token"][i - 1].split(".")[-1])
                    if tok in [">", "<", ">=", "<="] and \
                            sql_token["where"]["pre_token"][i - 1].split(".")[-1] not in range_col \
                            and sql_token["where"]["pre_token"][i - 1].split(".")[-1] in column_dict.keys():
                        range_col.append(sql_token["where"]["pre_token"][i - 1].split(".")[-1])
                else:
                    if tok == "=" and \
                            sql_token["where"]["pre_token"][i - 1] not in eq_col \
                            and sql_token["where"]["pre_token"][i - 1] in column_dict.keys():
                        eq_col.append(sql_token["where"]["pre_token"][i - 1])
                    if tok in [">", "<", ">=", "<="] and \
                            sql_token["where"]["pre_token"][i - 1] not in range_col \
                            and sql_token["where"]["pre_token"][i - 1] in column_dict.keys():
                        range_col.append(sql_token["where"]["pre_token"][i - 1])

        gro_col = list()
        if "group by" in sql_token.keys():
            for typ, tok in zip(sql_token["group by"]["pre_type"], sql_token["group by"]["pre_token"]):
                #  (0327): newly added. for job and real.
                if "tpch" in colinfo_load or "tpcds" in colinfo_load:
                    if typ == "group by#column" and tok.split(".")[-1] in column_dict.keys():
                        gro_col.append(tok.split(".")[-1])
                else:
                    if typ == "group by#column" and tok in column_dict.keys():
                        gro_col.append(tok)

        ord_col = list()
        if "order by" in sql_token.keys():
            for typ, tok in zip(sql_token["order by"]["pre_type"], sql_token["order by"]["pre_token"]):
                #  (0327): newly added. for job.
                if "tpch" in colinfo_load or "tpcds" in colinfo_load:
                    if typ == "order by#column" and tok.split(".")[-1] in column_dict.keys():
                        ord_col.append(tok.split(".")[-1])
                else:
                    if typ == "order by#column" and tok in column_dict.keys():
                        ord_col.append(tok)

        # 2. get the promising index combinations.
        # for col in join_col + eq_col + range_col:
        #     if tuple([column_dict[col]]) not in prom_ind[1]:
        #         prom_ind[1].append(tuple([column_dict[col]]))

        gro_tbl = list(set([col_info[col]["table"] for col in gro_col]))
        if len(gro_tbl) == 1:
            ind = tuple([column_dict[col] for col in gro_col])
            if len(ind) not in prom_ind.keys():
                prom_ind[len(ind)] = list()
            if ind not in prom_ind[len(ind)]:
                prom_ind[len(ind)].append(ind)
        ord_tbl = list(set([col_info[col]["table"] for col in ord_col]))
        if len(ord_tbl) == 1:
            ind = tuple([column_dict[col] for col in ord_col])
            if len(ind) not in prom_ind.keys():
                prom_ind[len(ind)] = list()
            if ind not in prom_ind[len(ind)]:
                prom_ind[len(ind)].append(ind)

        join_tbl = list()
        for col in join_col:
            tbl = col_info[col]["table"]
            if tbl in join_tbl:
                continue
            join_tbl.append(tbl)

            ind = tuple([column_dict[col] for col in join_col if col_info[col]['table'] == tbl])
            if len(ind) not in prom_ind.keys():
                prom_ind[len(ind)] = list()
            if ind not in prom_ind[len(ind)]:
                prom_ind[len(ind)].append(ind)

        j_EQ_r_tbl = list()
        for j in join_col:
            tbl = col_info[j]["table"]
            if tbl in j_EQ_r_tbl:
                continue

            # multiple join columns from the same table.
            j = [column_dict[col] for col in join_col if col_info[col]["table"] == tbl]
            EQ = [column_dict[col] for col in eq_col if col_info[col]["table"] == tbl]
            r = [column_dict[col] for col in range_col if col_info[col]["table"] == tbl]

            ind = tuple(j + EQ + r)
            if len(ind) not in prom_ind.keys():
                prom_ind[len(ind)] = list()
            if ind not in prom_ind[len(ind)]:
                prom_ind[len(ind)].append(ind)

        EQ_r_tbl = list()
        for EQ in eq_col:
            tbl = col_info[EQ]["table"]
            if tbl in EQ_r_tbl:
                continue

            r = [column_dict[col] for col in range_col if col_info[col]["table"] == tbl]

            ind = tuple([column_dict[EQ]] + r)
            if len(ind) not in prom_ind.keys():
                prom_ind[len(ind)] = list()
            if ind not in prom_ind[len(ind)]:
                prom_ind[len(ind)].append(ind)

        j_r_tbl = list()
        for j in join_col:
            tbl = col_info[j]["table"]
            if tbl in j_r_tbl:
                continue

            # multiple join columns from the same table.
            j = [column_dict[col] for col in join_col if col_info[col]["table"] == tbl]
            r = [column_dict[col] for col in range_col if col_info[col]["table"] == tbl]

            ind = tuple(j + r)
            if len(ind) not in prom_ind.keys():
                prom_ind[len(ind)] = list()
            if ind not in prom_ind[len(ind)]:
                prom_ind[len(ind)].append(ind)

        j_EQ_tbl = list()
        for j in join_col:
            tbl = col_info[j]["table"]
            if tbl in j_EQ_tbl:
                continue

            j = [column_dict[col] for col in join_col if col_info[col]["table"] == tbl]
            EQ = [column_dict[col] for col in eq_col if col_info[col]["table"] == tbl]

            ind = tuple(j + EQ)
            if len(ind) not in prom_ind.keys():
                prom_ind[len(ind)] = list()
            if ind not in prom_ind[len(ind)]:
                prom_ind[len(ind)].append(ind)

    for key in sorted(prom_ind.keys()):
        result_column_combinations.append(prom_ind[key])

    return result_column_combinations


def get_prom_index_candidates_from_pickle(token_load, colinfo_load, columns):
    result_column_combinations = list()

    with open(token_load, "rb") as rf:
        index_candidates = pickle.load(rf)

    column_dict = dict()
    for column in columns:
        column_dict[column.name] = column

    prom_ind = dict()
    for cand in index_candidates:
        ind_cols = cand.split("#")[-1].split(",")
        if len(ind_cols) not in prom_ind.keys():
            prom_ind[len(ind_cols)] = list()
        try:
            prom_ind[len(ind_cols)].append(tuple([column_dict[col] for col in ind_cols]))
        except Exception as e:
            logging.info("Some columns in the indexes are not involved in the global column set.")

    for key in sorted(prom_ind.keys()):
        result_column_combinations.append(prom_ind[key])

    return result_column_combinations


def get_prom_index_candidates_original(db_conf, query_texts, temp_load, columns):
    result_column_combinations = list()

    enc = en.encoding_schema(db_conf)
    parser = pi.Parser(enc["attr"])

    # (0822): newly added.
    workload_ori = list()
    for texts in query_texts:
        # (0917): newly modified.
        if isinstance(texts, list):
            workload_ori.extend(texts)
        if isinstance(texts, str):
            workload_ori.append(texts)

    # (0822): newly modified.
    if temp_load is not None:
        if temp_load.endswith(".pickle"):
            with open(temp_load, "rb") as rf:
                workload = pickle.load(rf)
        elif temp_load.endswith(".sql"):
            with open(temp_load, "r") as rf:
                workload = rf.readlines()
        # (0822): newly added.
        elif temp_load.endswith(".json"):
            with open(temp_load, "r") as rf:
                data = json.load(rf)

            workload = list()
            for item in data:
                if isinstance(item, dict):
                    if "workload" in item.keys():
                        workload.extend(item["workload"])
                    elif "sql" in item.keys():
                        workload.append(item["sql"])
                elif isinstance(item, list):
                    # (0822): newly modified. data:[item:[info:[]]]
                    if isinstance(item[0], list):
                        workload.extend([info[1] for info in item])
                    elif isinstance(item[0], str):
                        workload.extend(item)
                    elif isinstance(item[0], int):
                        workload.append(item[1])
                elif isinstance(item, str):
                    workload.append(item)

        workload_ori.extend(workload)

    workload_ori = list(set(workload_ori))

    for no, query in enumerate(workload_ori):
        b = psqlparse.parse_dict(query)
        parser.parse_stmt(b[0])
        parser.gain_candidates()
    index_candidates = parser.index_candidates
    index_candidates = list(index_candidates)
    index_candidates.sort()

    column_dict = dict()
    for column in columns:
        # (0822): newly modified.
        column_dict[f"{column.table}.{column.name}"] = column
        # column_dict[column.name] = column

    prom_ind = dict()
    for cand in index_candidates:
        # (0822): newly modified.
        tbl = cand.split("#")[0]
        ind_cols = cand.split("#")[-1].split(",")
        if len(ind_cols) not in prom_ind.keys():
            prom_ind[len(ind_cols)] = list()
        try:
            # (0822): newly modified.
            prom_ind[len(ind_cols)].append(tuple([column_dict[f"{tbl}.{col}"] for col in ind_cols]))
            # prom_ind[len(ind_cols)].append(tuple([column_dict[col] for col in ind_cols]))
        except Exception as e:
            logging.info("Some columns in the indexes are not involved in the global column set.")

    for key in sorted(prom_ind.keys()):
        # (0919): newly modified.
        result_column_combinations.append(sorted(list(set(prom_ind[key]))))

    return result_column_combinations


# (0917): newly added. index candidates generation by openGauss.
def get_prom_index_candidates_openGauss(db_conf, query_texts, temp_load, columns):
    result_column_combinations = list()

    col_dict = dict()
    for col in columns:
        col_dict[str(col)] = col

    # (0919): newly modified.
    db_conf_bak = copy.deepcopy(db_conf)

    db_conf_bak["postgresql"]["host"] = "xx.xx.xx.xx"
    db_conf_bak["postgresql"]["user"] = "xxxx"
    db_conf_bak["postgresql"]["password"] = "xxxx"
    db_conf_bak["postgresql"]["port"] = "xxxx"
    connector = PostgresDatabaseConnector(db_conf_bak, autocommit=True)

    workload_ori = list()
    for texts in query_texts:
        if isinstance(texts, list):
            workload_ori.extend(texts)
        if isinstance(texts, str):
            workload_ori.append(texts)

    if temp_load is not None:
        if temp_load.endswith(".pickle"):
            with open(temp_load, "rb") as rf:
                workload = pickle.load(rf)
        elif temp_load.endswith(".sql"):
            with open(temp_load, "r") as rf:
                workload = rf.readlines()
        elif temp_load.endswith(".json"):
            with open(temp_load, "r") as rf:
                data = json.load(rf)

            workload = list()
            for item in data:
                if isinstance(item, dict):
                    if "workload" in item.keys():
                        workload.extend(item["workload"])
                    elif "sql" in item.keys():
                        workload.append(item["sql"])
                elif isinstance(item, list):
                    if isinstance(item[0], list):
                        workload.extend([info[1] for info in item])
                    elif isinstance(item[0], str):
                        workload.extend(item)
                    elif isinstance(item[0], int):
                        workload.append(item[1])
                elif isinstance(item, str):
                    workload.append(item)

        workload_ori.extend(workload)

    workload_ori = list(set(workload_ori))

    api_query = "SELECT \"table\", \"column\" FROM gs_index_advise('{}')"

    prom_ind = dict()
    for query in workload_ori:
        # (0920): newly added.
        query = query.replace("2) ratio,", "2) AS ratio,")
        tbl_col = connector.exec_fetch(api_query.format(query.replace("'", "''")), one=False)

        for tbl, col in tbl_col:
            cs = col.split(",")
            if cs == [""]:
                continue

            if len(cs) not in prom_ind.keys():
                prom_ind[len(cs)] = list()

            try:
                prom_ind[len(cs)].append(tuple([col_dict[f"{tbl}.{c}"] for c in cs]))
            except Exception as e:
                logging.info("Some columns in the indexes are not involved in the global column set.")

    for key in sorted(prom_ind.keys()):
        # (0919): newly modified.
        result_column_combinations.append(sorted(list(set(prom_ind[key]))))

    return result_column_combinations


def create_column_permutation_indexes(columns, max_index_width):
    result_column_combinations = []

    table_column_dict = {}
    for column in columns:
        if column.table not in table_column_dict:
            table_column_dict[column.table] = set()
        table_column_dict[column.table].add(column)

    for length in range(1, max_index_width + 1):
        unique = set()
        count = 0
        for key, columns_per_table in table_column_dict.items():
            unique |= set(itertools.permutations(columns_per_table, length))  # permutation: the orders that matter.
            count += len(set(itertools.permutations(columns_per_table, length)))
        logging.info(f"the total number of the unique {length}-column indexes is: {count}")

        # (1212): newly added, sorted list.
        result_column_combinations.append(sorted(list(unique)))
        # result_column_combinations.append(list(unique))

    return result_column_combinations


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


def read_row_query(sql_list, columns):
    workload = list()
    for query_id, query_text in enumerate(sql_list):
        # exp_conf, type="template"
        # if type == "template" and exp_conf["queries"] \
        #         and query_id + 1 not in exp_conf["queries"]:
        #     continue

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

            query_details[query] = {
                "cost_without_indexes": cost_without_indexes,
                "cost_with_indexes": cost_with_indexes,
                "utilized_indexes": utilized_indexes_query,
            }

    return utilized_indexes_workload, query_details


# Storage
def b_to_mb(b):
    return b / 1000 / 1000


def mb_to_b(mb):
    return mb * 1000 * 1000


# Time
def s_to_ms(s):
    return s * 1000
