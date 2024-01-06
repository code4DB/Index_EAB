# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: dqn_run
# @Author: Wei Zhou
# @Time: 2023/8/16 16:15

import os
import time
import pickle
import logging

import Model as model

from index_advisor_selector.index_selection.dqn_selection.dqn_utils import Encoding as en
from index_advisor_selector.index_selection.dqn_selection.dqn_utils import ParserForIndex as pi
from index_advisor_selector.index_selection.dqn_selection.dqn_utils.Common import get_parser, set_logger, gen_cands

conf = {"LR": 0.002, "EPSILON": 0.97, "Q_ITERATION": 200, "U_ITERATION": 5, "BATCH_SIZE": 64,
        "GAMMA": 0.95, "EPISODES": 1000, "LEARNING_START": 600, "DECAY_EP": 50, "MEMORY_CAPACITY": 20000}


def train_dqn(args):
    conf["NAME"] = args.exp_id
    conf["EPISODES"] = args.epoch

    time_start = time.time()

    index_mode = "hypo"
    if not os.path.exists(os.path.dirname(args.logdir.format(args.exp_id))):
        os.makedirs(os.path.dirname(args.logdir.format(args.exp_id)))
    if not os.path.exists(os.path.dirname(args.model_save.format(args.exp_id, 0))):
        os.makedirs(os.path.dirname(args.model_save.format(args.exp_id, 0)))
    set_logger(args.runlog.format(args.exp_id))

    logging.info(f"Load workload from `{args.work_load}`.")

    if args.work_load.endswith(".pickle"):
        with open(args.work_load, "rb") as rf:
            workload = pickle.load(rf)
    elif args.work_load.endswith(".sql"):
        with open(args.work_load, "r") as rf:
            workload = rf.readlines()
    frequency = [1 for _ in range(len(workload))]

    if os.path.exists(args.cand_load):
        logging.info(f"Load candidate from `{args.cand_load}`.")
        with open(args.cand_load, "rb") as rf:
            index_candidates = pickle.load(rf)
    else:
        logging.info(f"Generate candidate based on `{args.work_load}`.")
        enc = en.encoding_schema(args.conf_load)
        sql_parser = pi.Parser(enc["attr"])

        index_candidates = gen_cands(workload, sql_parser)

    agent = model.DQN(args, workload, frequency, index_candidates, index_mode,
                      conf, args.is_dnn, args.is_ps, args.is_double, args.a, args.action_mode)
    best_indexes, last_indexes = agent.train()

    time_end = time.time()

    indexes = list()
    for _i, _idx in enumerate(last_indexes):
        if _idx == 1.0:
            indexes.append(index_candidates[_i])

    no_cost, ind_cost = list(), list()
    total_no_cost, total_ind_cost = 0, 0

    agent.envx.pg_client2.delete_indexes()
    for query in workload:
        cost = agent.envx.pg_client2.get_queries_cost([query])[0]
        no_cost.append(cost)
        total_no_cost += cost

    for index in indexes:
        agent.envx.pg_client2.execute_create_hypo(index)
    for query in workload:
        cost = agent.envx.pg_client2.get_queries_cost([query])[0]
        ind_cost.append(cost)
        total_ind_cost += cost
    agent.envx.pg_client2.delete_indexes()

    data = {"workload": workload,
            "indexes": indexes,
            "no_cost": no_cost,
            "total_no_cost": total_no_cost,
            "ind_cost": ind_cost,
            "total_ind_cost": total_ind_cost,
            "sel_info": {"time_duration": time_end - time_start}}
    return data


def get_dqn_res(args):
    time_start = time.time()

    index_mode = "hypo"

    if args.work_load.endswith(".pickle"):
        with open(args.work_load, "rb") as rf:
            workload = pickle.load(rf)
    elif args.work_load.endswith(".sql"):
        with open(args.work_load, "r") as rf:
            workload = rf.readlines()
    frequency = [1 for _ in range(len(workload))]

    if os.path.exists(args.cand_load):
        with open(args.cand_load, "rb") as rf:
            index_candidates = pickle.load(rf)
    else:
        enc = en.encoding_schema(args.conf_load)
        sql_parser = pi.Parser(enc["attr"])

        index_candidates = gen_cands(workload, sql_parser)

    agent = model.DQN(args, workload, frequency, index_candidates, index_mode,
                      conf, args.is_dnn, args.is_ps, args.is_double, args.a, args.action_mode)
    _indexes = agent.infer()

    time_end = time.time()

    indexes = list()
    for _i, _idx in enumerate(_indexes):
        if _idx == 1.0:
            indexes.append(index_candidates[_i])

    no_cost, ind_cost = list(), list()
    total_no_cost, total_ind_cost = 0, 0

    agent.envx.pg_client2.delete_indexes()
    for query in workload:
        cost = agent.envx.pg_client2.get_queries_cost([query])[0]
        no_cost.append(cost)
        total_no_cost += cost

    for index in indexes:
        agent.envx.pg_client2.execute_create_hypo(index)
    for query in workload:
        cost = agent.envx.pg_client2.get_queries_cost([query])[0]
        ind_cost.append(cost)
        total_ind_cost += cost
    agent.envx.pg_client2.delete_indexes()

    data = {"workload": workload,
            "indexes": indexes,
            "no_cost": no_cost,
            "total_no_cost": total_no_cost,
            "ind_cost": ind_cost,
            "total_ind_cost": total_ind_cost,
            "sel_info": {"time_duration": time_end - time_start}}
    return data


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # (0813): newly added.
    logging.disable(logging.DEBUG)

    if args.action_mode == "train":
        train_dqn(args)
    elif args.action_mode == "infer":
        data = get_dqn_res(args)
        print(data)
