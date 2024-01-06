# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: mcts_run
# @Author: Wei Zhou
# @Time: 2023/8/16 15:49

import os
import json
import pickle
import logging
import configparser

from index_advisor_selector.index_selection.mcts_selection.mcts_advisor import MCTSAdvisor
from index_advisor_selector.index_selection.mcts_selection.mcts_utils.mcts_com import get_parser, pre_work, set_logger
from index_advisor_selector.index_selection.mcts_selection.mcts_utils.cost_evaluation import CostEvaluation
from index_advisor_selector.index_selection.mcts_selection.mcts_utils.postgres_dbms import PostgresDatabaseConnector
from index_advisor_selector.index_selection.mcts_selection.mcts_utils.mcts_workload import Index


class MCTSEncoder(json.JSONEncoder):
    def default(self, obj):
        # 👇️ if passed in object is instance of Decimal
        # convert it to a string

        if isinstance(obj, Index):
            return str(obj)
        if isinstance(obj, set):
            return list(obj)
        if "Index" in str(obj.__class__):
            return str(obj)

        # 👇️ otherwise use the default behavior
        return json.JSONEncoder.default(self, obj)


def get_mcts_res(args, work_list):
    if not os.path.exists(os.path.dirname(args.log_file.format(args.exp_id))):
        os.makedirs(os.path.dirname(args.log_file.format(args.exp_id)))

    # (1018): to be removed. (uncommented)
    # set_logger(args.log_file.format(args.exp_id))

    db_conf = configparser.ConfigParser()
    db_conf.read(args.db_file)

    # (1030): newly added.
    if args.db_name is not None:
        db_conf["postgresql"]["database"] = args.db_name

    database_connector = PostgresDatabaseConnector(db_conf, autocommit=True)

    # (0818): newly added.
    if os.path.exists(args.model_load):
        with open(args.model_load, "rb") as rf:
            advisor = pickle.load(rf)
        advisor.did_run = False

        advisor.database_connector = database_connector
        advisor.cost_evaluation = CostEvaluation(database_connector)
        advisor.mcts_tree.pg_utils = database_connector
        advisor.mcts_tree.cost_evaluation = CostEvaluation(database_connector)

    else:
        advisor = MCTSAdvisor(database_connector, args, process=args.process)

    workload = pre_work(work_list, args.schema_file, args.varying_frequencies)
    indexes, sel_infos = advisor.calculate_best_indexes(workload, overhead=args.overhead)

    advisor.database_connector = None
    advisor.cost_evaluation = None
    advisor.mcts_tree.pg_utils = None
    advisor.mcts_tree.cost_evaluation = None
    with open(args.model_save.format(args.exp_id), "wb") as wf:
        pickle.dump(advisor, wf)

    indexes_pre = list()
    for index in indexes:
        index_pre = f"{index.columns[0].table.name}#{','.join([col.name for col in index.columns])}"
        indexes_pre.append(index_pre)
    indexes_pre.sort()

    no_cost, ind_cost = list(), list()
    total_no_cost, total_ind_cost = 0, 0

    freq_list = list()
    for query in workload:
        cost = database_connector.get_ind_cost(query.text, "", mode="hypo") * query.frequency
        total_no_cost += cost
        no_cost.append(cost)

        cost = database_connector.get_ind_cost(query.text, indexes_pre, mode="hypo") * query.frequency
        total_ind_cost += cost
        ind_cost.append(cost)

        freq_list.append(query.frequency)

    data = {"config": vars(args),
            "workload": [work_list, freq_list],
            "indexes": indexes_pre,
            "no_cost": no_cost,
            "total_no_cost": total_no_cost,
            "ind_cost": ind_cost,
            "total_ind_cost": total_ind_cost,
            "sel_info": sel_infos}

    return data


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    data = get_mcts_res(args)
