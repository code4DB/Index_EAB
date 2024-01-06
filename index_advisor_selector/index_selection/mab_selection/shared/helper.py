import os
import json
import logging
from random import random

import psqlparse
import configparser

import numpy as np
import pandas as pd

import seaborn as sns
from pandas import DataFrame
import matplotlib.pyplot as plt

import constants


def plot_histogram(statistics_dict, title, experiment_id):
    """
    Simple plot function to plot the average reward, and a line to show the best possible reward

    :param statistics_dict: list of statistic histograms
    :param title: title of the plot
    :param experiment_id: id of the current experiment
    """
    for statistic_name, statistic_histogram in statistics_dict.items():
        plt.plot(statistic_histogram, label=statistic_name)
    plt.title(title)
    plt.xlabel("Rounds")
    plt.legend()
    plt.savefig(get_experiment_folder_path(experiment_id) + title + ".png")
    plt.show()


def plot_histogram_v2(statistics_dict, window_size, title, experiment_id):
    """
    Simple plot function to plot the average reward, and a line to show the best possible reward

    :param statistics_dict: list of statistic histograms
    :param window_size: size of the moving window
    :param title: title of the plot
    :param experiment_id: id of the current experiment
    """
    crop_amount = int(np.ceil(window_size / 2))
    for statistic_name, statistic_histogram in statistics_dict.items():
        plt.plot(statistic_histogram[crop_amount:-crop_amount], label=statistic_name)
    plt.title(title)
    plt.xlabel("Rounds")
    plt.legend()
    plt.savefig(get_experiment_folder_path(experiment_id) + title + ".png")
    plt.show()


def get_experiment_folder_path(experiment_id):
    """
    Get the folder location of the experiment
    :param experiment_id: name of the experiment
    :return: file path as string
    """
    experiment_folder_path = f"{constants.ROOT_DIR}/{constants.EXPERIMENT_FOLDER}/{experiment_id}/"
    if not os.path.exists(experiment_folder_path):
        os.makedirs(experiment_folder_path)
    if not os.path.exists(f"{experiment_folder_path}/model"):
        os.makedirs(f"{experiment_folder_path}/model")

    return experiment_folder_path


def get_log_folder_path(experiment_id):
    # (0816): newly added.
    log_folder_path = f"{constants.ROOT_DIR}/{constants.EXPERIMENT_FOLDER}/{experiment_id}/logdir"
    if not os.path.exists(log_folder_path):
        os.makedirs(log_folder_path)

    return log_folder_path


def get_workload_folder_path(experiment_id):
    """
    Get the folder location of the experiment
    :param experiment_id: name of the experiment
    :return: file path as string
    """
    experiment_folder_path = f"{constants.ROOT_DIR}/{constants.WORKLOADS_FOLDER}/{experiment_id}"
    if not os.path.exists(experiment_folder_path):
        os.makedirs(experiment_folder_path)
    return experiment_folder_path


def plot_histogram_avg(statistic_dict, title, experiment_id):
    """
    Simple plot function to plot the average reward, and a line to show the best possible reward
    :param statistic_dict: list of statistic histograms
    :param title: title of the plot
    :param experiment_id: id of the current experiment
    """
    for statistic_name, statistic_list in statistic_dict.items():
        for i in range(1, len(statistic_list)):
            statistic_list[i] = statistic_list[i - 1] + statistic_list[i]
            statistic_list[i - 1] = statistic_list[i - 1] / i
        statistic_list[len(statistic_list) - 1] = statistic_list[len(statistic_list) - 1] / len(statistic_list)

    plot_histogram(statistic_dict, title, experiment_id)


def plot_moving_average(statistic_dict, window_size, title, experiment_id):
    """
    Simple plot function to plot the moving average of a histograms
    :param statistic_dict: list of statistic histograms
    :param window_size: size of the moving window
    :param title: title of the plot
    :param experiment_id: id of the current experiment
    """
    statistic_avg_dict = {}
    for statistic_name, statistic_list in statistic_dict.items():
        avg_mask = np.ones(window_size) / window_size
        statistic_list_avg = np.convolve(statistic_list, avg_mask, "same")
        statistic_avg_dict[statistic_name] = statistic_list_avg

    plot_histogram_v2(statistic_avg_dict, window_size, title, experiment_id)


def parse(parsed_res, item_res, tbl_col):
    if "fromClause" not in parsed_res.keys():
        for key in parsed_res.keys():
            if isinstance(parsed_res[key], dict):
                item_res = parse(parsed_res[key], item_res, tbl_col)
            else:
                return item_res
    else:
        table_alias = dict()
        for tbl_info in parsed_res["fromClause"]:
            if "RangeSubselect" in tbl_info.keys():
                item_res = parse(tbl_info["RangeSubselect"]["subquery"]["SelectStmt"], item_res, tbl_col)
            elif "JoinExpr" in tbl_info.keys():
                for key in tbl_info["JoinExpr"].keys():
                    if isinstance(tbl_info["JoinExpr"][key], dict) and \
                            "RangeVar" in tbl_info["JoinExpr"][key].keys():
                        tbl_name = tbl_info["JoinExpr"][key]["RangeVar"]["relname"]
                        if tbl_name not in table_alias.keys():
                            table_alias[tbl_name] = list()

                        if "alias" in tbl_info["JoinExpr"][key]["RangeVar"].keys():
                            alias = tbl_info["JoinExpr"][key]["RangeVar"]["alias"]["Alias"]["aliasname"]
                            if alias not in table_alias[tbl_name]:
                                table_alias[tbl_name].append(alias)

                for tbl in table_alias.keys():
                    if tbl not in tbl_col.keys():
                        continue
                    for col in tbl_col[tbl]:
                        if len(table_alias[tbl]) != 0:
                            for alias in table_alias[tbl]:
                                col_ref = "{'ColumnRef': {'fields': [{'String': {'str': '" + alias + "'}}, {'String': {'str': '" + col + "'}}"
                                if col_ref in str(tbl_info["JoinExpr"][key]):
                                    if tbl.upper() not in item_res["predicates"].keys():
                                        item_res["predicates"][tbl.upper()] = list()
                                    item_res["predicates"][tbl.upper()].append(col.upper())
                                    break
                        else:
                            col_ref = "{'ColumnRef': {'fields': [{'String': {'str': '" + col + "'}}"
                            if col_ref in str(tbl_info["JoinExpr"][key]):
                                if tbl.upper() not in item_res["predicates"].keys():
                                    item_res["predicates"][tbl.upper()] = list()
                                item_res["predicates"][tbl.upper()].append(col.upper())
            elif "RangeVar" in tbl_info.keys():
                tbl_name = tbl_info["RangeVar"]["relname"]

                if tbl_name not in table_alias.keys():
                    table_alias[tbl_name] = list()

                # alias = tbl_name
                if "alias" in tbl_info["RangeVar"].keys():
                    alias = tbl_info["RangeVar"]["alias"]["Alias"]["aliasname"]
                    if alias not in table_alias[tbl_name]:
                        table_alias[tbl_name].append(alias)

        if "withClause" in parsed_res.keys():
            for cte in parsed_res["withClause"]["WithClause"]["ctes"]:
                if cte["CommonTableExpr"]["ctename"] in table_alias.keys():
                    table_alias.pop(cte["CommonTableExpr"]["ctename"])
                item_res = parse(cte["CommonTableExpr"]["ctequery"]["SelectStmt"], item_res, tbl_col)

        clause_map = {"targetList": "payload",
                      "whereClause": "predicates",
                      "groupClause": "group_by",
                      "sortClause": "order_by"}

        for clause in clause_map.keys():
            if clause not in parsed_res.keys():
                continue

            for tbl in table_alias.keys():
                if tbl not in tbl_col.keys():
                    continue
                for col in tbl_col[tbl]:
                    if len(table_alias[tbl]) != 0:
                        for alias in table_alias[tbl]:
                            col_ref = "{'ColumnRef': {'fields': [{'String': {'str': '" + alias + "'}}, {'String': {'str': '" + col + "'}}"
                            if col_ref in str(parsed_res[clause]):
                                if tbl.upper() not in item_res[clause_map[clause]].keys():
                                    item_res[clause_map[clause]][tbl.upper()] = list()
                                item_res[clause_map[clause]][tbl.upper()].append(col.upper())
                                break
                    else:
                        col_ref = "{'ColumnRef': {'fields': [{'String': {'str': '" + col + "'}}"
                        if col_ref in str(parsed_res[clause]):
                            if tbl.upper() not in item_res[clause_map[clause]].keys():
                                item_res[clause_map[clause]][tbl.upper()] = list()
                            item_res[clause_map[clause]][tbl.upper()].append(col.upper())

        return item_res


def parse_other(parsed_res, item_res, tbl_col):
    table_alias = dict()
    tbl_info = parsed_res["relation"]
    if "RangeSubselect" in tbl_info.keys():
        item_res = parse(tbl_info["RangeSubselect"]["subquery"]["SelectStmt"], item_res, tbl_col)
    elif "JoinExpr" in tbl_info.keys():
        for key in tbl_info["JoinExpr"].keys():
            if isinstance(tbl_info["JoinExpr"][key], dict) and \
                    "RangeVar" in tbl_info["JoinExpr"][key].keys():
                tbl_name = tbl_info["JoinExpr"][key]["RangeVar"]["relname"]
                if tbl_name not in table_alias.keys():
                    table_alias[tbl_name] = list()

                if "alias" in tbl_info["JoinExpr"][key]["RangeVar"].keys():
                    alias = tbl_info["JoinExpr"][key]["RangeVar"]["alias"]["Alias"]["aliasname"]
                    if alias not in table_alias[tbl_name]:
                        table_alias[tbl_name].append(alias)

        for tbl in table_alias.keys():
            if tbl not in tbl_col.keys():
                continue
            for col in tbl_col[tbl]:
                if len(table_alias[tbl]) != 0:
                    for alias in table_alias[tbl]:
                        col_ref = "{'ColumnRef': {'fields': [{'String': {'str': '" + alias + "'}}, {'String': {'str': '" + col + "'}}"
                        if col_ref in str(tbl_info["JoinExpr"][key]):
                            if tbl.upper() not in item_res["predicates"].keys():
                                item_res["predicates"][tbl.upper()] = list()
                            item_res["predicates"][tbl.upper()].append(col.upper())
                            break
                else:
                    col_ref = "{'ColumnRef': {'fields': [{'String': {'str': '" + col + "'}}"
                    if col_ref in str(tbl_info["JoinExpr"][key]):
                        if tbl.upper() not in item_res["predicates"].keys():
                            item_res["predicates"][tbl.upper()] = list()
                        item_res["predicates"][tbl.upper()].append(col.upper())
    else:
        tbl_name = tbl_info["RangeVar"]["relname"]
        if tbl_name not in table_alias.keys():
            table_alias[tbl_name] = list()

        # alias = tbl_name
        if "alias" in tbl_info["RangeVar"].keys():
            alias = tbl_info["RangeVar"]["alias"]["Alias"]["aliasname"]
            if alias not in table_alias[tbl_name]:
                table_alias[tbl_name].append(alias)

    if "withClause" in parsed_res.keys():
        for cte in parsed_res["withClause"]["WithClause"]["ctes"]:
            if cte["CommonTableExpr"]["ctename"] in table_alias.keys():
                table_alias.pop(cte["CommonTableExpr"]["ctename"])
            item_res = parse(cte["CommonTableExpr"]["ctequery"]["SelectStmt"], item_res, tbl_col)

    clause_map = {"targetList": "payload",
                  "whereClause": "predicates",
                  "groupClause": "group_by",
                  "sortClause": "order_by"}

    for clause in clause_map.keys():
        if clause not in parsed_res.keys():
            continue

        for tbl in table_alias.keys():
            if tbl not in tbl_col.keys():
                continue
            for col in tbl_col[tbl]:
                if len(table_alias[tbl]) != 0:
                    for alias in table_alias[tbl]:
                        col_ref = "{'ColumnRef': {'fields': [{'String': {'str': '" + alias + "'}}, {'String': {'str': '" + col + "'}}"
                        if col_ref in str(parsed_res[clause]):
                            if tbl.upper() not in item_res[clause_map[clause]].keys():
                                item_res[clause_map[clause]][tbl.upper()] = list()
                            item_res[clause_map[clause]][tbl.upper()].append(col.upper())
                            break
                else:
                    col_ref = "{'ColumnRef': {'fields': [{'String': {'str': '" + col + "'}}"
                    if col_ref in str(parsed_res[clause]):
                        if tbl.upper() not in item_res[clause_map[clause]].keys():
                            item_res[clause_map[clause]][tbl.upper()] = list()
                        item_res[clause_map[clause]][tbl.upper()].append(col.upper())

    return item_res


def parse_column(work_data, schema_load, varying_frequencies=False):
    # work_load = "/data/wz/index/index_eab/eab_olap/bench_temp/tpch_template_18.sql"
    # schema_load = "/data/wz/index/attack/data_resource/db_info/schema_tpch_1gb.json"

    # if work_load.endswith(".sql"):
    #     with open(work_load, "r") as rf:
    #         work_data = rf.readlines()
    # elif work_load.endswith(".json"):
    #     with open(work_load, "r") as rf:
    #         work_data = json.load(rf)
    #     work_list = list()
    #     for typ in work_data.keys():
    #         for queries in work_data[typ].values():
    #             work_list.append(queries[0])
    #     work_data = work_list

    with open(schema_load, "r") as rf:
        schema_info = json.load(rf)

    tbl_col = dict()
    for info in schema_info:
        tbl_col[info["table"]] = [col["name"] for col in info["columns"]]

    all_res = list()
    for no, sql in enumerate(work_data):
        # (1016): newly modified.
        if isinstance(sql, list):
            item_res = {"id": sql[0], "query_string": sql[1], "predicates": {}, "payload": {}, "group_by": {},
                        "order_by": {}}
        else:
            item_res = {"id": no, "query_string": sql, "predicates": {}, "payload": {}, "group_by": {}, "order_by": {}}

        if varying_frequencies:
            if isinstance(sql, list):
                item_res["freq"] = sql[-1]
            else:
                item_res["freq"] = random.randint(1, 1000)
        else:
            item_res["freq"] = 1

        parsed_res = psqlparse.parse_dict(item_res["query_string"])[0]
        # parsed_res = psqlparse.parse_dict(item_res["query_string"])[0]["SelectStmt"]

        # (1009): newly added.
        if "SelectStmt" in parsed_res.keys():
            parsed_res = parsed_res["SelectStmt"]
            item_res = parse(parsed_res, item_res, tbl_col)

        elif "UpdateStmt" in parsed_res.keys():
            parsed_res = parsed_res["UpdateStmt"]
            item_res = parse_other(parsed_res, item_res, tbl_col)

        elif "DeleteStmt" in parsed_res.keys():
            parsed_res = parsed_res["DeleteStmt"]
            item_res = parse_other(parsed_res, item_res, tbl_col)

        else:
            raise NameError


        all_res.append(item_res)

    # logging.info(f"Parse the workload from `{work_load}`.")

    return all_res


def get_queries_v2(workload, schema_file, varying_frequencies=False):
    """
    Read all the queries in the queries pointed by the QUERY_DICT_FILE constant
    :return: list of queries
    """
    # # Reading the configuration for given experiment ID
    # exp_config = configparser.ConfigParser()
    # exp_config.read(constants.ROOT_DIR + constants.EXPERIMENT_CONFIG)
    #
    # # experiment id for the current run
    # experiment_id = exp_config["general"]["run_experiment"]
    # workload_file = str(exp_config[experiment_id]["workload_file"])

    # with open(workload_file, "r") as rf:
    #     queries = json.load(rf)

    queries = parse_column(workload, schema_file, varying_frequencies)

    # logging.info(f"Load workload from `{workload_file}`.")

    return queries


def get_normalized(value, assumed_min, assumed_max, history_list):
    """
    This method gives a normalized reward based on the reward history

    :param value: current reward that we need to normalize
    :param history_list: rewards we got up to now, including the current reward
    :param assumed_min: assumed minimum value
    :param assumed_max: assumed maximum value
    :return: normalized reward (0 - 1)
    """
    if len(history_list) > 5:
        real_min = min(history_list) - 1
        real_max = max(history_list)
    else:
        real_min = min(min(history_list), assumed_min)
        real_max = max(max(history_list), assumed_max)
    return (value - real_min) / (real_max - real_min)


def update_dict_list(current, new):
    """
    This function does merging operation of 2 dictionaries with lists as values. This method adds only new values found
    in the new list to the old list

    :param current: current list
    :param new: new list
    :return: merged list
    """
    for table, predicates in new.items():
        if table not in current:
            current[table] = predicates
        else:
            temp_set = set(new[table]) - set(current[table])
            current[table] = current[table] + list(temp_set)
    return current


def plot_exp_report(exp_id, exp_report_list, measurement_names, log_y=False):
    """
    Creates a plot for several experiment reports
    :param exp_id: ID of the experiment
    :param exp_report_list: This can contain several exp report objects
    :param measurement_names: What measurement that we will use for y
    :param log_y: draw y axis in log scale
    """
    for measurement_name in measurement_names:
        comps = list()
        final_df = DataFrame()
        for exp_report in exp_report_list:
            df = exp_report.data
            df[constants.DF_COL_COMP_ID] = exp_report.component_id
            final_df = pd.concat([final_df, df])
            comps.append(exp_report.component_id)

        final_df = final_df[final_df[constants.DF_COL_MEASURE_NAME] == measurement_name]
        # Error style = "band" / "bars"
        # sns_plot = sns.relplot(x=constants.DF_COL_BATCH, y=constants.DF_COL_MEASURE_VALUE, hue=constants.DF_COL_COMP_ID,
        #                        kind="line", ci="sd", data=final_df, err_style="band")

        # (0814): newly modified.
        sns_plot = sns.relplot(x=constants.DF_COL_BATCH, y=constants.DF_COL_MEASURE_VALUE, hue=constants.DF_COL_COMP_ID,
                               kind="line", errorbar="sd", data=final_df, err_style="band")

        if log_y:
            sns_plot.set(yscale="log")
        plot_title = measurement_name + " Comparison"
        sns_plot.set(xlabel=constants.DF_COL_BATCH, ylabel=measurement_name)
        sns_plot.savefig(get_experiment_folder_path(exp_id) + plot_title + ".png")


def create_comparison_tables(exp_id, exp_report_list):
    """
    Create a CSV with numbers that are important for the comparison

    :param exp_id: ID of the experiment
    :param exp_report_list: This can contain several exp report objects
    :return:
    """
    final_df = DataFrame(
        columns=[constants.DF_COL_COMP_ID, constants.DF_COL_BATCH_COUNT, constants.MEASURE_HYP_BATCH_TIME,
                 constants.MEASURE_INDEX_RECOMMENDATION_COST, constants.MEASURE_INDEX_CREATION_COST,
                 constants.MEASURE_QUERY_EXECUTION_COST, constants.MEASURE_TOTAL_WORKLOAD_TIME])

    for exp_report in exp_report_list:
        data = exp_report.data
        component = exp_report.component_id
        rounds = exp_report.batches_per_rep
        reps = exp_report.reps

        # Get information from the data frame
        hyp_batch_time = get_avg_measure_value(data, constants.MEASURE_HYP_BATCH_TIME, reps)
        recommend_time = get_avg_measure_value(data, constants.MEASURE_INDEX_RECOMMENDATION_COST, reps)
        creation_time = get_avg_measure_value(data, constants.MEASURE_INDEX_CREATION_COST, reps)
        elapsed_time = get_avg_measure_value(data, constants.MEASURE_QUERY_EXECUTION_COST, reps)
        total_workload_time = get_avg_measure_value(data, constants.MEASURE_BATCH_TIME, reps) + hyp_batch_time

        # Adding to the final data frame
        final_df.loc[len(final_df)] = [component, rounds, hyp_batch_time, recommend_time, creation_time, elapsed_time,
                                       total_workload_time]

    final_df.round(4).to_csv(get_experiment_folder_path(exp_id) + "comparison_table.csv")


#  - remove min and max
def get_avg_measure_value(data, measure_name, reps):
    return (data[data[constants.DF_COL_MEASURE_NAME] == measure_name][constants.DF_COL_MEASURE_VALUE].sum()) / reps


def get_sum_measure_value(data, measure_name):
    return data[data[constants.DF_COL_MEASURE_NAME] == measure_name][constants.DF_COL_MEASURE_VALUE].sum()


def change_experiment(args):
    """
    Programmatically change the experiment
    """
    exp_config = configparser.ConfigParser()
    exp_config.read(args.exp_file)
    exp_config["general"]["run_experiment"] = args.bench

    for key in vars(args):
        if key in exp_config[args.bench]:
            exp_config[args.bench][key] = str(args.__getattribute__(key))

    with open(args.exp_file.replace(".conf", "_5.conf"), "w") as configfile:
        exp_config.write(configfile)


def log_configs(logging, module):
    for variable in dir(module):
        if not variable.startswith("__"):
            logging.info(str(variable) + ": " + str(getattr(module, variable)))
