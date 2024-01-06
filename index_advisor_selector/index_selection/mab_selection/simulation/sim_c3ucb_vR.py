import json

import numpy
from pandas import DataFrame
from torch.utils.tensorboard import SummaryWriter

import os
import datetime
import logging
import operator
import pprint
from importlib import reload

import constants as constants
from database import sql_connection as sql_connection
from database import sql_helper_v2 as sql_helper
from shared import configs_v2 as configs
from shared import helper as helper

from bandits import bandit_c3ucb_v2 as bandits
from bandits import bandit_helper_v2 as bandit_helper
from bandits.experiment_report import ExpReport
from bandits.oracle_v2 import OracleV7 as Oracle
from bandits.query_v5 import Query
from shared.helper import get_log_folder_path
from shared.mab_com import plot_report


# Simulation built on vQ to collect the super arm performance
class BaseSimulator:
    def __init__(self, args, workload):
        # (1016): newly added.
        self.args = args

        # configuring the logger
        # log_file = f"{helper.get_experiment_folder_path(configs.experiment_id)}/{configs.experiment_id}.log"
        # logging.basicConfig(
        #     filename=log_file,
        #     filemode="w", format="%(asctime)s - %(levelname)s - %(message)s")
        # logging.getLogger().setLevel(logging.INFO)

        # Get the query List
        # 'id', 'query_string', 'predicates', 'payload', 'group_by', 'order_by'

        # (1016): newly modified.
        # with open(args.workload_file, "r") as rf:
        #     workload = json.load(rf)

        self.queries = helper.get_queries_v2(workload, args.schema_file, args.varying_frequencies)
        self.connection = sql_connection.get_sql_connection(args, args.db_file)
        self.query_obj_store = {}
        reload(bandit_helper)

        # (0816): newly added.
        self.exp_id = args.exp_id
        self.writer = SummaryWriter(get_log_folder_path(args.exp_id))

        self.measure = {"Workload Cost": list(),
                        "Index Utility": list(),
                        "Total Time": list()}

        self.params_load = args.params_load
        self.params_save = args.params_save
        self.save_gap = args.save_gap

        # (0818): newly added.
        self.process = args.process


class Simulator(BaseSimulator):
    def run(self):
        pp = pprint.PrettyPrinter()
        reload(configs)
        results = list()
        super_arm_scores = dict()
        super_arm_counts = dict()
        best_super_arm = set()
        logging.info("Logging configs...\n")
        helper.log_configs(logging, configs)
        logging.info("Logging constants...\n")
        helper.log_configs(logging, constants)
        logging.info("Starting MAB...\n")

        # Get all the columns from the database
        all_columns, number_of_columns = sql_helper.get_all_columns(self.connection)
        context_size = number_of_columns * (
                1 + constants.CONTEXT_UNIQUENESS + constants.CONTEXT_INCLUDES) + constants.STATIC_CONTEXT_SIZE

        # Create oracle and the bandit
        # (0814): newly modified.
        # configs.max_memory -= int(sql_helper.get_current_pds_size(self.connection))
        oracle = Oracle(configs.constraint, configs.max_memory, configs.max_count, self.process)
        c3ucb_bandit = bandits.C3UCB(context_size, configs.input_alpha, configs.input_lambda, oracle)
        # (0818): newly added.
        if os.path.exists(self.params_load):
            c3ucb_bandit.load_params(self.params_load)

        # Running the bandit for T rounds and gather the reward
        arm_selection_count = dict()
        chosen_arms_last_round = dict()
        next_workload_shift = 0
        queries_start = configs.queries_start_list[next_workload_shift]
        queries_end = configs.queries_end_list[next_workload_shift]
        query_obj_additions = list()
        total_time = 0.0

        total_start_time_round = datetime.datetime.now()
        for t in range(configs.rounds):
            sql_helper.get_current_index(self.connection)

            # for t in range((configs.rounds + configs.hyp_rounds)):
            logging.info(f"round: {t}")
            start_time_round = datetime.datetime.now()
            # At the start of the round we will read the applicable set for the current round.
            # This is a workaround used to demo the dynamic query flow.
            # We read the queries from the start and move the window each round.

            # # New set of queries in this batch, required for query execution
            # queries_current_batch = self.queries[queries_start:queries_end]

            # (0814): newly added.
            if t == 0:
                # New set of queries in this batch, required for query execution
                queries_current_batch = self.queries[queries_start:queries_end]
                work_list = [query["query_string"] for query in self.queries[queries_start:queries_end]]

                no_cost = list()
                execute_cost_no_index = 0
                for query in queries_current_batch:
                    # (1016): newly modified. `* query["freq"]`
                    time = sql_helper.hyp_execute_query(self.connection, query["query_string"]) * query["freq"]
                    no_cost.append(time)
                    execute_cost_no_index += time

            # check if workload shift is required. todo: ?
            # if t == configs.workload_shifts[next_workload_shift]:
            # if t - configs.hyp_rounds == configs.workload_shifts[next_workload_shift]:
            #     queries_start = configs.queries_start_list[next_workload_shift]
            #     queries_end = configs.queries_end_list[next_workload_shift]
            #     if len(configs.workload_shifts) > next_workload_shift + 1:
            #         next_workload_shift += 1
            #
            #     # New set of queries in this batch, required for query execution
            #     queries_current_batch = self.queries[queries_start:queries_end]
            #     work_list = [query["query_string"] for query in self.queries[queries_start:queries_end]]
            #
            #     # (0814): newly added. no index?
            #     no_cost = list()
            #     execute_cost_no_index = 0
            #     for query in queries_current_batch:
            #         time = sql_helper.hyp_execute_query(self.connection, query["query_string"])
            #         no_cost.append(time)
            #         execute_cost_no_index += time

            # Adding new queries to the query store
            query_obj_list_current = list()
            for n in range(len(queries_current_batch)):
                query = queries_current_batch[n]
                query_id = query["id"]
                if query_id in self.query_obj_store:
                    query_obj_in_store = self.query_obj_store[query_id]
                    query_obj_in_store.frequency += 1
                    query_obj_in_store.last_seen = t
                    query_obj_in_store.query_string = query["query_string"]
                    if query_obj_in_store.first_seen == -1:
                        query_obj_in_store.first_seen = t
                else:
                    query = Query(self.connection, query_id, query["query_string"],
                                  query["predicates"], query["payload"], t, freq=query["freq"])
                    # (0815): for what?
                    query.context = bandit_helper.get_query_context_v1(query, all_columns, number_of_columns)
                    self.query_obj_store[query_id] = query
                query_obj_list_current.append(self.query_obj_store[query_id])

            # This list contains all past queries, we don't include new queries seen for the first time.
            query_obj_list_past = list()
            query_obj_list_new = list()
            for key, obj in self.query_obj_store.items():
                if t - obj.last_seen <= constants.QUERY_MEMORY and 0 <= obj.first_seen < t:
                    query_obj_list_past.append(obj)
                elif t - obj.last_seen > constants.QUERY_MEMORY:
                    obj.first_seen = -1
                elif obj.first_seen == t:
                    query_obj_list_new.append(obj)

            # We don't want to reset in the first round,
            # If there is new additions or removals, We identify a workload change.
            if t > 0 and len(query_obj_additions) > 0:
                workload_change = len(query_obj_additions) / len(query_obj_list_past)
                # 1) workload_change > 0.5, reset the parameters;
                # 2) workload_change <= 0.5, forget_factor (1 - workload_change * 2), params * forget_factor.
                c3ucb_bandit.workload_change_trigger(workload_change)

            # This rounds new will be the additions for the next round
            query_obj_additions = query_obj_list_new

            # indexes = sql_helper.get_current_index(self.connection)
            # logging.info(f"L160, The list of the current indexes ({len(indexes)}) is: {indexes}.")

            # Get the predicates for queries and Generate index arms for each query
            index_arms = dict()
            for i in range(len(query_obj_list_past)):
                bandit_arms_tmp = bandit_helper.gen_arms_from_predicates_v2(self.connection, query_obj_list_past[i])
                for key, index_arm in bandit_arms_tmp.items():
                    if key not in index_arms:
                        index_arm.query_ids = set()
                        index_arm.query_ids_backup = set()
                        index_arm.clustered_index_time = 0
                        index_arms[key] = index_arm
                    index_arm.clustered_index_time += max(
                        query_obj_list_past[i].table_scan_times[index_arm.table_name]) if \
                        query_obj_list_past[i].table_scan_times[index_arm.table_name] else 0
                    index_arms[key].query_ids.add(index_arm.query_id)
                    index_arms[key].query_ids_backup.add(index_arm.query_id)

            # indexes = sql_helper.get_current_index(self.connection)
            # logging.info(f"L179, The list of the current indexes ({len(indexes)}) is: {indexes}.")

            # Set the index arms at the bandit
            if t == configs.hyp_rounds and configs.hyp_rounds != 0:
                index_arms = dict()
            index_arm_list = list(index_arms.values())
            logging.info(f"Generated `{len(index_arm_list)}` arms")
            c3ucb_bandit.set_arms(index_arm_list)

            # indexes = sql_helper.get_current_index(self.connection)
            # logging.info(f"L189, The list of the current indexes ({len(indexes)}) is: {indexes}.")

            # creating the context, here we pass all the columns in the database
            # index column & column position
            context_vectors_v1 = bandit_helper.get_name_encode_context_vectors_v2(index_arms, all_columns,
                                                                                  number_of_columns,
                                                                                  constants.CONTEXT_UNIQUENESS,
                                                                                  constants.CONTEXT_INCLUDES)

            # indexes = sql_helper.get_current_index(self.connection)
            # logging.info(f"L199, The list of the current indexes ({len(indexes)}) is: {indexes}.")

            # index_usage_last_batch, index_size / database_size, is_include
            context_vectors_v2 = bandit_helper.get_derived_value_context_vectors_v3(self.connection, index_arms,
                                                                                    query_obj_list_past,
                                                                                    chosen_arms_last_round,
                                                                                    not constants.CONTEXT_INCLUDES)

            # indexes = sql_helper.get_current_index(self.connection)
            # logging.info(f"L208, The list of the current indexes ({len(indexes)}) is: {indexes}.")

            context_vectors = list()
            for i in range(len(context_vectors_v1)):
                context_vectors.append(
                    numpy.array(list(context_vectors_v2[i]) + list(context_vectors_v1[i]),
                                ndmin=2))

            # getting the super arm from the bandit
            chosen_arm_ids = c3ucb_bandit.select_arm_v2(context_vectors, t == configs.rounds - 1)
            if t >= configs.hyp_rounds and t - configs.hyp_rounds > constants.STOP_EXPLORATION_ROUND:
                chosen_arm_ids = list(best_super_arm)

            # get objects for the chosen set of arm ids
            chosen_arms = dict()
            used_memory = 0
            if chosen_arm_ids:
                chosen_arms = dict()
                for arm in chosen_arm_ids:
                    index_name = index_arm_list[arm].index_name
                    chosen_arms[index_name] = index_arm_list[arm]
                    used_memory = used_memory + index_arm_list[arm].memory
                    if index_name in arm_selection_count:
                        arm_selection_count[index_name] += 1
                    else:
                        arm_selection_count[index_name] = 1

            # clean everything at the start of actual rounds
            if configs.hyp_rounds != 0 and t == configs.hyp_rounds:
                sql_helper.bulk_drop_index(self.connection, constants.SCHEMA_NAME, chosen_arms_last_round)
                chosen_arms_last_round = {}

            # finding the difference between last round and this round
            keys_last_round = set(chosen_arms_last_round.keys())
            keys_this_round = set(chosen_arms.keys())
            key_intersection = keys_last_round & keys_this_round
            key_additions = keys_this_round - key_intersection
            key_deletions = keys_last_round - key_intersection
            logging.info(f"Selected: {keys_this_round}")
            logging.debug(f"Added: {key_additions}")
            logging.debug(f"Removed: {key_deletions}")

            added_arms = dict()
            deleted_arms = dict()
            for key in key_additions:
                added_arms[key] = chosen_arms[key]
            for key in key_deletions:
                deleted_arms[key] = chosen_arms_last_round[key]

            start_time_create_query = datetime.datetime.now()

            # indexes = sql_helper.get_current_index(self.connection)
            # logging.info(f"L260, The list of the current indexes ({len(indexes)}) is: {indexes}.")

            # : newly added. `execute_cost_no_index`. execute_cost, creation_cost, arm_rewards
            time_split, time_taken, creation_cost_dict, arm_rewards = sql_helper.hyp_create_query_drop_new(
                self.connection,
                constants.SCHEMA_NAME,
                chosen_arms,
                added_arms,
                deleted_arms,
                query_obj_list_current,
                execute_cost_no_index)
            # indexes = sql_helper.get_current_index(self.connection)
            # logging.info(f"L271, The list of the current indexes ({len(indexes)}) is: {indexes}.")

            end_time_create_query = datetime.datetime.now()

            creation_cost = sum(creation_cost_dict.values())
            if t == configs.hyp_rounds and configs.hyp_rounds != 0:
                # logging arm usage counts
                logging.info("\n\nIndex Usage Counts:\n" + pp.pformat(
                    sorted(arm_selection_count.items(), key=operator.itemgetter(1), reverse=True)))
                arm_selection_count = {}

            c3ucb_bandit.update_v4(chosen_arm_ids, arm_rewards)

            # (0818): newly added.
            # if t % self.save_gap == 0:
            #     c3ucb_bandit.save_params(self.params_save.format(self.exp_id, str(t)))

            super_arm_id = frozenset(chosen_arm_ids)
            if t >= configs.hyp_rounds:
                if super_arm_id in super_arm_scores:
                    super_arm_scores[super_arm_id] = super_arm_scores[super_arm_id] * super_arm_counts[super_arm_id] \
                                                     + time_taken
                    super_arm_counts[super_arm_id] += 1
                    super_arm_scores[super_arm_id] /= super_arm_counts[super_arm_id]
                else:
                    super_arm_counts[super_arm_id] = 1
                    super_arm_scores[super_arm_id] = time_taken

            # keeping track of queries that we saw last time
            chosen_arms_last_round = chosen_arms

            # (0815): last round?
            if t == (configs.rounds + configs.hyp_rounds - 1):
                sql_helper.bulk_drop_index(self.connection, constants.SCHEMA_NAME, chosen_arms)

            end_time_round = datetime.datetime.now()

            # (0816): newly added. `Index Utility`
            self.measure["Workload Cost"].append(time_taken)
            self.measure["Index Utility"].append(1 - time_taken / execute_cost_no_index)
            self.measure["Total Time"].append((end_time_round - start_time_round).total_seconds())
            # self.writer.add_scalar("episode_reward", time_taken, t)

            if t > self.args.min_rounds:
                num = numpy.sum(numpy.array(self.measure["Workload Cost"])[-self.args.early_stopping:]
                                == self.measure["Workload Cost"][-1])
                if num == self.args.early_stopping:
                    break

            # current_config_size = float(sql_helper.get_current_pds_size(self.connection))
            # logging.info("Size taken by the config: " + str(current_config_size) + "MB")
            # (0814): newly modified.
            logging.info(f"Size taken by the config: {used_memory} MB")

            time_duration = (end_time_round - start_time_round).total_seconds()

            # Adding information to the results array
            if t >= configs.hyp_rounds:
                actual_round_number = t - configs.hyp_rounds
                recommendation_time = (end_time_round - start_time_round).total_seconds() - (
                        end_time_create_query - start_time_create_query).total_seconds()
                total_round_time = creation_cost + time_taken + recommendation_time
                results.append([actual_round_number, constants.MEASURE_BATCH_TIME, total_round_time])
                results.append([actual_round_number, constants.MEASURE_INDEX_CREATION_COST, creation_cost])
                results.append([actual_round_number, constants.MEASURE_QUERY_EXECUTION_COST, time_taken])
                results.append(
                    [actual_round_number, constants.MEASURE_INDEX_RECOMMENDATION_COST, recommendation_time])
                # results.append([actual_round_number, constants.MEASURE_MEMORY_COST, current_config_size])
                # (0814): newly modified.
                results.append([actual_round_number, constants.MEASURE_MEMORY_COST, used_memory])
            else:
                total_round_time = (end_time_round - start_time_round).total_seconds() - (
                        end_time_create_query - start_time_create_query).total_seconds()
                results.append([t, constants.MEASURE_HYP_BATCH_TIME, total_round_time])
            total_time += total_round_time

            if t >= configs.hyp_rounds:
                best_super_arm = min(super_arm_scores, key=super_arm_scores.get)

            # print(f"Time taken by bandit for {t} rounds: {time_taken}.")
            # print(f"current total {t}: ", total_time)
            # print(f"Index arms selected by bandit for {t} rounds: \n"
            #       f"{sorted([arm.index_cols for arm in chosen_arms.values()])}.")
        total_end_time_round = datetime.datetime.now()

        total_time_duration = (total_end_time_round - total_start_time_round).total_seconds()

        # plot_report(self.exp_id, self.measure)

        logging.info("Time taken by bandit for " + str(configs.rounds) + " rounds: " + str(total_time))
        logging.info("\n\nIndex Usage Counts:\n" + pp.pformat(
            sorted(arm_selection_count.items(), key=operator.itemgetter(1), reverse=True)))

        # return results, total_time
        indexes = list()
        for index in chosen_arms.values():
            index_pre = f"{index.table_name.lower()}#{','.join(index.index_cols).lower()}"
            indexes.append(index_pre)
        indexes.sort()

        freq_list = [query["freq"] for query in self.queries]

        data = {"config": vars(self.args),
                "workload": [work_list, freq_list],
                "indexes": indexes,
                "no_cost": no_cost,
                "total_no_cost": execute_cost_no_index,
                "ind_cost": time_split,
                "total_ind_cost": time_taken,
                "sel_info": {"infer_time_duration": time_duration,
                             "time_duration": total_time_duration}}

        # indexes
        return data


if __name__ == "__main__":
    # Running MAB
    exp_report_mab = ExpReport(configs.experiment_id, constants.COMPONENT_MAB, configs.reps, configs.rounds)
    for r in range(configs.reps):
        simulator = Simulator()
        sim_results, total_workload_time = simulator.run()
        temp = DataFrame(sim_results, columns=[constants.DF_COL_BATCH, constants.DF_COL_MEASURE_NAME,
                                               constants.DF_COL_MEASURE_VALUE])
        temp.append([-1, constants.MEASURE_TOTAL_WORKLOAD_TIME, total_workload_time])
        temp[constants.DF_COL_REP] = r
        exp_report_mab.add_data_list(temp)

    # plot line graphs
    helper.plot_exp_report(configs.experiment_id, [exp_report_mab],
                           (constants.MEASURE_BATCH_TIME, constants.MEASURE_QUERY_EXECUTION_COST))
