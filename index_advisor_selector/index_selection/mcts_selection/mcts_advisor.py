# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: mcts_advisor
# @Author: Wei Zhou
# @Time: 2023/7/21 17:31

import os
import copy
import json
import time
import logging
import configparser

from index_advisor_selector.index_selection.heu_selection.heu_utils.heu_com import get_utilized_indexes, get_columns_from_schema
from index_advisor_selector.index_selection.heu_selection.heu_utils.candidate_generation import candidates_per_query, \
    syntactically_relevant_indexes_dqn_rule, \
    syntactically_relevant_indexes_openGauss

from .mcts_model import State, Node, MCTS

from .mcts_utils.cost_evaluation import CostEvaluation
from .mcts_utils.postgres_dbms import PostgresDatabaseConnector
from .mcts_utils.mcts_com import syntactically_relevant_indexes, get_parser, pre_work, plot_report, mb_to_b
from .mcts_utils.mcts_workload import Column, Table, Index, Workload


class MCTSAdvisor:
    def __init__(self, database_connector, parameters, process=False):
        self.did_run = False

        self.parameters = parameters

        self.database_connector = database_connector
        self.database_connector.drop_indexes()
        self.cost_evaluation = CostEvaluation(database_connector)

        self.mcts_tree = None

        # : newly added. for process visualization.
        self.process = process
        self.step = {"selected": list()}

    def calculate_best_indexes(self, workload, overhead=False):
        assert self.did_run is False, "Selection algorithm can only run once."
        self.did_run = True

        estimation_num_bef = self.database_connector.cost_estimations
        estimation_duration_bef = self.database_connector.cost_estimation_duration

        simulation_num_bef = self.database_connector.simulated_indexes
        simulation_duration_bef = self.database_connector.index_simulation_duration

        time_start = time.time()
        indexes = self._calculate_best_indexes(workload)
        time_end = time.time()

        estimation_duration_aft = self.database_connector.cost_estimation_duration
        estimation_num_aft = self.database_connector.cost_estimations

        simulation_num_aft = self.database_connector.simulated_indexes
        simulation_duration_aft = self.database_connector.index_simulation_duration

        # : newly added. for selection runtime
        cache_hits = self.cost_evaluation.cache_hits
        cost_requests = self.cost_evaluation.cost_requests

        self.cost_evaluation.complete_cost_estimation()

        # : newly added.
        if self.process:
            if overhead:
                return indexes, {"step": self.step, "cache_hits": cache_hits,
                                 "cost_requests": cost_requests, "time_duration": time_end - time_start,
                                 "estimation_num": estimation_num_aft - estimation_num_bef,
                                 "estimation_duration": estimation_duration_aft - estimation_duration_bef,
                                 "simulation_num": simulation_num_aft - simulation_num_bef,
                                 "simulation_duration": simulation_duration_aft - simulation_duration_bef}
            else:
                return indexes, {"step": self.step, "cache_hits": cache_hits, "cost_requests": cost_requests}
        elif overhead:
            return indexes, {"cache_hits": cache_hits, "cost_requests": cost_requests,
                             "time_duration": time_end - time_start,
                             "estimation_num": estimation_num_aft - estimation_num_bef,
                             "estimation_duration": estimation_duration_aft - estimation_duration_bef,
                             "simulation_num": simulation_num_aft - simulation_num_bef,
                             "simulation_duration": simulation_duration_aft - simulation_duration_bef}
        else:
            return indexes, ""

    def _calculate_best_indexes(self, workload):
        """
        :param workload:
        :return:
        """
        logging.info("Calculating the best indexes by MCTS")

        # 1. synthesize the potential index candidate set.
        # potential_index = syntactically_relevant_indexes(workload, self.parameters.max_index_width)

        # Generate syntactically relevant candidates
        # (0917): newly added.
        if self.parameters.cand_gen is None or self.parameters.cand_gen == "permutation":
            potential_index = candidates_per_query(
                Workload(workload),
                self.parameters.max_index_width,
                candidate_generator=syntactically_relevant_indexes,
            )

        elif self.parameters.cand_gen == "dqn_rule":
            db_conf = configparser.ConfigParser()
            db_conf.read(self.parameters.db_file)

            _, columns = get_columns_from_schema(self.parameters.schema_file)

            potential_index = [syntactically_relevant_indexes_dqn_rule(db_conf, [query.text], columns,
                                                                       self.parameters.max_index_width) for query in
                               workload]

        elif self.parameters.cand_gen == "openGauss":
            db_conf = configparser.ConfigParser()
            db_conf.read(self.parameters.db_file)

            _, columns = get_columns_from_schema(self.parameters.schema_file)

            potential_index = [syntactically_relevant_indexes_openGauss(db_conf, [query.text], columns,
                                                                        self.parameters.max_index_width) for query in
                               workload]

        # (0918): newly modified.
        if self.parameters.cand_gen is None or self.parameters.is_utilized:
            # Obtain the utilized indexes considering every single query
            potential_index, _ = get_utilized_indexes(Workload(workload), potential_index, self.cost_evaluation)
        else:
            cand_set = list()
            for cand in potential_index:
                cand_set.extend(cand)
            candidates = set(cand_set)

            potential_index = copy.deepcopy(candidates)

            _ = self.cost_evaluation.calculate_cost(
                Workload(workload), potential_index
                , store_size=True  # newly added.
            )

        potential_index = sorted(potential_index)

        # 1. synthesize the potential index candidate set.
        # if not self.parameters.is_utilized:
        #     potential_index = syntactically_relevant_indexes(workload, self.parameters.max_index_width)
        # else:
        #     candidates = candidates_per_query(
        #         Workload(workload),
        #         self.parameters.max_index_width,
        #         candidate_generator=syntactically_relevant_indexes,
        #     )
        #
        #     potential_index, _ = get_utilized_indexes(
        #         Workload(workload), candidates, self.cost_evaluation, True
        #     )
        #
        #     potential_index = sorted(potential_index)

        # _ = self.cost_evaluation.calculate_cost(Workload(workload), potential_index, store_size=True)

        # (0805): newly added. for `storage`.
        if self.parameters.constraint == "storage":
            # _ = self.cost_evaluation.calculate_cost(Workload(workload), potential_index, store_size=True)

            potential_index_filter = list()
            for index in potential_index:
                if index.estimated_size <= mb_to_b(self.parameters.storage):
                    potential_index_filter.append(index)
            potential_index = copy.deepcopy(potential_index_filter)

            # potential_index_pre = list()
            # for index in potential_index:
            #     tbl, col = index.split("#")
            #     col = [Column(c, Table(tbl)) for c in col.split(",")]
            #     potential_index_pre.append(Index(col))
            # _ = self.cost_evaluation.calculate_cost(Workload(workload), potential_index_pre, store_size=True)

            # potential_index_pre_filter = list()
            # for index1, index2 in zip(potential_index, potential_index_pre):
            #     if index2.estimated_size <= mb_to_b(self.parameters.storage):
            #         potential_index_pre_filter.append(index1)
            # potential_index = copy.deepcopy(potential_index_pre_filter)

        # 2. index selection based on MCTS.
        current_index = list()

        root = Node(State(current_index, potential_index, self.parameters.constraint,
                          self.parameters.cardinality, self.parameters.storage))
        # (0818): newly added.
        if self.mcts_tree is None:
            self.mcts_tree = MCTS(self.parameters, workload, potential_index,
                                  self.database_connector, self.cost_evaluation, self.process)
        final_conf, final_reward = self.mcts_tree.mcts_search(self.parameters.budget, root)

        # (0818): newly added.
        save_dir = os.path.dirname(self.parameters.log_file.format(self.parameters.exp_id))
        # plot_report(save_dir, self.mcts_tree.measure)

        if self.process:
            self.step = copy.deepcopy(self.mcts_tree.step)

        return final_conf


if __name__ == "__main__":
    process, overhead = True, True

    parser = get_parser()
    args = parser.parse_args()

    # bench = "tpch"
    # if bench == "tpch":  # 1
    #     args.schema_file = "../data_resource/db_info/schema_tpch_1gb.json"
    #     args.work_file = "../data_resource/bench_template/tpch_template21_index_info.json"
    #     args.work_file = "../data_resource/visrew_spj_data/server103/tpch_1gb/s103_tpch_1gb_4all05_tblcol_index_all.json"
    #     args.work_file = "../data_resource/visrew_spj_data/bak/server151/tpch_1gb/s151_tpch_1gb_4all01_tblcol_index_all.json"
    #     args.db_file = "../data_resource/db_info/tpch_conf/db103_tpch_1gb.conf"
    #
    # with open(args.work_file, "r") as rf:
    #     work_json = json.load(rf)
    # work_list = [work_json[0]["sql"]]
    #
    # args.budget = 200
    # args.cardinality = 5
    # args.max_index_width = 2
    # args.select_policy = "UCT"  # ["UCT", "EPSILON"]
    # args.roll_num = 1
    # args.best_policy = "BG"  # ["BCE", "BG"]

    db_conf = configparser.ConfigParser()
    db_conf.read(args.db_file)
    database_connector = PostgresDatabaseConnector(db_conf, autocommit=True)
    advisor = MCTSAdvisor(database_connector, args, process=process)

    with open(args.work_file, "r") as rf:
        work_list = rf.readlines()

    workload = pre_work(work_list, args.schema_file)
    indexes, sel_infos = advisor.calculate_best_indexes(workload, overhead=overhead)

    no_cost, ind_cost = 0, 0
    for sql in work_list:
        no_cost += database_connector.get_ind_cost(sql, "", mode="hypo")
        ind_cost += database_connector.get_ind_cost(sql, indexes, mode="hypo")
