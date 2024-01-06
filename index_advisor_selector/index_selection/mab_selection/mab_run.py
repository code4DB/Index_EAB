# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: mab_run
# @Author: Wei Zhou
# @Time: 2023/8/14 10:12

import pickle
from pandas import DataFrame

import logging
from importlib import reload

import constants
from bandits.experiment_report import ExpReport
from shared.mab_com import get_parser, set_logger
from shared import configs_v2 as configs, helper


# Define Experiment ID list that we need to run
# tpc_h_static_10_MAB, tpc_h_skew_static_10_MAB, tpc_ds_static_10_MAB, tpc_ds_random_10_MAB

def main(args):
    # Generate form saved reports
    FROM_FILE = False
    PLOT_LOG_Y = False
    PLOT_MEASURE = (constants.MEASURE_BATCH_TIME, constants.MEASURE_QUERY_EXECUTION_COST,
                    constants.MEASURE_INDEX_CREATION_COST)

    exp_report_list = list()
    # f"{constants.ROOT_DIR}/{constants.EXPERIMENT_FOLDER}/{experiment_id}/"
    experiment_folder_path = helper.get_experiment_folder_path(args.exp_id)
    helper.change_experiment(args)
    reload(configs)
    reload(logging)

    # configuring the logger
    if not FROM_FILE:
        # logging.basicConfig(
        #     filename=f"{experiment_folder_path}/{configs.experiment_id}.log",
        #     filemode="w", format="%(asctime)s - %(levelname)s - %(message)s")

        filename = f"{experiment_folder_path}/{args.exp_id}.log"
        set_logger(filename)
        logging.getLogger().setLevel(constants.LOGGING_LEVEL)

    if FROM_FILE:
        with open(f"{experiment_folder_path}/reports.pickle", "rb") as f:
            exp_report_list = exp_report_list + pickle.load(f)
        logging.info(f"Load reports from: `{experiment_folder_path}/reports.pickle`")
    else:
        logging.info(f"Currently running: {args.exp_id}")

        Simulators = {}
        for mab_version in configs.mab_versions:
            Simulators[mab_version] = (getattr(__import__(mab_version, fromlist=["Simulator"]), "Simulator"))
        for version, Simulator in Simulators.items():
            version_number = version.split("_v", 1)[1]
            exp_report_mab = ExpReport(args.exp_id,
                                       constants.COMPONENT_MAB + version_number + args.exp_id,
                                       configs.reps, configs.rounds)
            # 1. read the queries:
            # 'id', 'query_string', 'predicates', 'payload', 'group_by', 'order_by';
            # 2. set up the connection.
            simulator = Simulator(args)
            results, total_workload_time = simulator.run()
            temp = DataFrame(results, columns=[constants.DF_COL_BATCH, constants.DF_COL_MEASURE_NAME,
                                               constants.DF_COL_MEASURE_VALUE])
            temp.append([-1, constants.MEASURE_TOTAL_WORKLOAD_TIME, total_workload_time])
            exp_report_mab.add_data_list(temp)
            exp_report_list.append(exp_report_mab)

        # Save results
        with open(f"{experiment_folder_path}/reports.pickle", "wb") as f:
            pickle.dump(exp_report_list, f)

    helper.plot_exp_report(args.exp_id, exp_report_list, PLOT_MEASURE, PLOT_LOG_Y)
    helper.create_comparison_tables(args.exp_id, exp_report_list)


def get_mab_res(args, workload):
    # f"{constants.ROOT_DIR}/{constants.EXPERIMENT_FOLDER}/{experiment_id}/"
    experiment_folder_path = helper.get_experiment_folder_path(args.exp_id)
    helper.change_experiment(args)
    reload(configs)
    reload(logging)

    # configuring the log
    # (1018): to be removed. (uncommented)
    # filename = f"{experiment_folder_path}/{args.exp_id}.log"
    # set_logger(filename)
    # logging.getLogger().setLevel(constants.LOGGING_LEVEL)
    # logging.info(f"Currently running: {args.exp_id}")

    Simulators = dict()
    for mab_version in configs.mab_versions:
        Simulators[mab_version] = (getattr(__import__(mab_version, fromlist=["Simulator"]), "Simulator"))
    for version, Simulator in Simulators.items():
        # 1. read the queries:
        # 'id', 'query_string', 'predicates', 'payload', 'group_by', 'order_by';
        # 2. set up the connection.
        simulator = Simulator(args, workload)

        # sel_info
        data = simulator.run()
        return data


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    data = get_mab_res(args)
