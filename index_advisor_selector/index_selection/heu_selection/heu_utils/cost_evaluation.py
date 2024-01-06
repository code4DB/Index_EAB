import numpy as np

import logging

from .workload import Workload
from .what_if_index_creation import WhatIfIndexCreation

from index_advisor_selector.index_benefit_estimation.tree_model.tree_cost_infer import load_model_tree, get_tree_est_res
from index_advisor_selector.index_benefit_estimation.index_cost_lib.lib_infer import load_model_lib, get_lib_est_res
from index_advisor_selector.index_benefit_estimation.query_former.former_infer import load_model_former, get_former_est_res


class CostEvaluation:
    def __init__(self, db_connector, cost_estimation="whatif"):
        logging.debug("Init cost evaluation")
        self.db_connector = db_connector
        self.cost_estimation = cost_estimation
        logging.info("Cost estimation with " + self.cost_estimation)
        self.what_if = WhatIfIndexCreation(db_connector)
        self.current_indexes = set()
        self.cost_requests = 0
        self.cache_hits = 0
        # Cache structure:
        # {(query_object, relevant_indexes): cost}
        self.cache = {}
        self.completed = False
        # It is not necessary to drop hypothetical indexes during __init__().
        # These are only created per connection. Hence, non should be present.

        self.relevant_indexes_cache = {}

        # self.model = load_model_tree()
        # self.model = load_model_lib()
        # self.model = load_model_former()

    def estimate_size(self, index):
        # : Refactor: It is currently too complicated to compute
        # We must search in current indexes to get an index object with .hypopg_oid
        result = None
        for i in self.current_indexes:
            if index == i:
                result = i
                break
        if result:
            # Index does currently exist and size can be queried
            if not index.estimated_size:
                index.estimated_size = self.what_if.estimate_index_size(result.hypopg_oid)
        else:
            self._simulate_or_create_index(index, store_size=True)

    def which_indexes_utilized_and_cost(self, query, indexes):
        # simulate hypothetical indexes all together.
        self._prepare_cost_calculation(indexes, store_size=True)

        plan = self.db_connector.get_plan(query)

        # (0917): newly modified.
        # cost = plan["Total Cost"] * query.frequency
        # (1008): newly modified.
        cost = self.calculate_cost(Workload([query]), indexes)

        # (1014): newly added.
        self._prepare_cost_calculation(indexes, store_size=True)
        plan = self.db_connector.get_plan(query)

        plan_str = str(plan)

        recommended_indexes = set()

        # We are iterating over the CostEvaluation's indexes and not over `indexes`
        # because it is not guaranteed that hypopg_name is set for all items in
        # `indexes`. This is caused by _prepare_cost_calculation that only creates
        # indexes which are not yet existing. If there is no hypothetical index
        # created for an index object, there is no hypopg_name assigned to it. However,
        # all items in current_indexes must also have an equivalent in `indexes`.
        for index in self.current_indexes:
            assert (
                    index in indexes
            ), "Something went wrong with _prepare_cost_calculation."

            if index.hypopg_name not in plan_str:
                continue
            recommended_indexes.add(index)

        return recommended_indexes, cost

    def calculate_cost(self, workload, indexes, store_size=False):
        # calculate_cost
        assert (
                self.completed is False
        ), "Cost Evaluation is completed and cannot be reused."
        self._prepare_cost_calculation(indexes, store_size=store_size)
        total_cost = 0

        # : Make query cost higher for queries which are running often
        for query in workload.queries:
            self.cost_requests += 1
            # total_cost += self._request_cache(query, indexes)
            # (0824): newly modified.
            total_cost += self._request_cache(query, indexes) * query.frequency

            # if "insert" in query.text.lower():
            #     total_cost += 0 * query.frequency
            # else:
            #     total_cost += self._request_cache(query, indexes) * query.frequency

        return total_cost

    def calculate_cost_without_interaction(self, workload, indexes, store_size=False):
        # calculate_cost_without_interaction
        # without index interaction
        assert (
                self.completed is False
        ), "Cost Evaluation is completed and cannot be reused."

        # : Make query cost higher for queries which are running often
        if len(indexes) != 0:
            total_cost = 0
            for index in indexes:
                self._prepare_cost_calculation([index], store_size=store_size)

                for query in workload.queries:
                    self.cost_requests += 1
                    # total_cost += self._request_cache(query, indexes)
                    # (0824): newly modified.
                    total_cost += self._request_cache(query, indexes) * query.frequency
            total_cost = total_cost / len(indexes)
        else:
            self._prepare_cost_calculation([], store_size=store_size)

            total_cost = 0
            for query in workload.queries:
                self.cost_requests += 1
                # total_cost += self._request_cache(query, indexes)
                # (0824): newly modified.
                total_cost += self._request_cache(query, indexes) * query.frequency
        return total_cost

    def calculate_cost_tree(self, workload, indexes, store_size=False):
        # calculate_cost_tree
        # learned index benefit estimation
        assert (
                self.completed is False
        ), "Cost Evaluation is completed and cannot be reused."
        self._prepare_cost_calculation(indexes, store_size=store_size)
        total_cost = 0

        # : Make query cost higher for queries which are running often
        for query in workload.queries:
            self.cost_requests += 1

            q_i_hash = (query, frozenset(indexes))
            if q_i_hash in self.relevant_indexes_cache:
                relevant_indexes = self.relevant_indexes_cache[q_i_hash]
            else:
                relevant_indexes = self._relevant_indexes(query, indexes)
                self.relevant_indexes_cache[q_i_hash] = relevant_indexes

            # Check if query and corresponding relevant indexes in cache
            if (query, relevant_indexes) in self.cache:
                self.cache_hits += 1
                cost = self.cache[(query, relevant_indexes)]
            # If no cache hit request cost from database system
            else:
                plan = self.db_connector.get_plan(query)
                cost = float(np.exp(get_tree_est_res(self.model, plan)))

                self.cache[(query, relevant_indexes)] = cost

            total_cost += cost * query.frequency

        return total_cost

    def calculate_cost_lib(self, workload, indexes, store_size=False):
        # calculate_cost_lib
        # learned index benefit estimation
        assert (
                self.completed is False
        ), "Cost Evaluation is completed and cannot be reused."
        self._prepare_cost_calculation(indexes, store_size=store_size)
        self._prepare_cost_calculation(set(), store_size=store_size)

        total_cost = 0

        # : Make query cost higher for queries which are running often
        for query in workload.queries:
            self.cost_requests += 1

            q_i_hash = (query, frozenset(indexes))
            if q_i_hash in self.relevant_indexes_cache:
                relevant_indexes = self.relevant_indexes_cache[q_i_hash]
            else:
                relevant_indexes = self._relevant_indexes(query, indexes)
                self.relevant_indexes_cache[q_i_hash] = relevant_indexes

            # Check if query and corresponding relevant indexes in cache
            if (query, relevant_indexes) in self.cache:
                self.cache_hits += 1
                cost = self.cache[(query, relevant_indexes)]
            # If no cache hit request cost from database system
            else:
                plan = self.db_connector.get_plan(query)
                if len(indexes) == 0:
                    cost = 1. * plan["Total Cost"]
                else:
                    indexes_temp = [str(ind) for ind in indexes]
                    cols = [ind.split(",") for ind in indexes_temp]
                    cols = [list(map(lambda x: x.split(".")[-1], col)) for col in cols]
                    indexes_temp = [f"{ind.split('.')[0]}#{','.join(col)}" for ind, col in zip(indexes_temp, cols)]

                    cost = get_lib_est_res(self.model, indexes_temp, plan)[0] * plan["Total Cost"]

                self.cache[(query, relevant_indexes)] = cost

            total_cost += cost * query.frequency

        return total_cost

    def calculate_cost_former(self, workload, indexes, store_size=False):
        # calculate_cost_former
        # learned index benefit estimation
        assert (
                self.completed is False
        ), "Cost Evaluation is completed and cannot be reused."
        self._prepare_cost_calculation(indexes, store_size=store_size)
        total_cost = 0

        # : Make query cost higher for queries which are running often
        for query in workload.queries:
            self.cost_requests += 1

            q_i_hash = (query, frozenset(indexes))
            if q_i_hash in self.relevant_indexes_cache:
                relevant_indexes = self.relevant_indexes_cache[q_i_hash]
            else:
                relevant_indexes = self._relevant_indexes(query, indexes)
                self.relevant_indexes_cache[q_i_hash] = relevant_indexes

            # Check if query and corresponding relevant indexes in cache
            if (query, relevant_indexes) in self.cache:
                self.cache_hits += 1
                cost = self.cache[(query, relevant_indexes)]
            # If no cache hit request cost from database system
            else:
                plan = self.db_connector.get_plan(query)
                data = [{"w/ plan": plan, "w/ actual cost": 666}]
                cost = get_former_est_res(self.model, data)[0]

                self.cache[(query, relevant_indexes)] = cost

            total_cost += cost * query.frequency

        return total_cost

    # Creates the current index combination by simulating/creating
    # missing indexes and unsimulating/dropping indexes
    # that exist but are not in the combination.
    def _prepare_cost_calculation(self, indexes, store_size=False):
        for index in set(indexes) - self.current_indexes:
            self._simulate_or_create_index(index, store_size=store_size)
        for index in self.current_indexes - set(indexes):
            self._unsimulate_or_drop_index(index)

        assert self.current_indexes == set(indexes)

    def _simulate_or_create_index(self, index, store_size=False):
        if self.cost_estimation == "whatif":
            self.what_if.simulate_index(index, store_size=store_size)
        elif self.cost_estimation == "actual_runtimes":
            self.db_connector.create_index(index)
        self.current_indexes.add(index)

    def _unsimulate_or_drop_index(self, index):
        if self.cost_estimation == "whatif":
            self.what_if.drop_simulated_index(index)
        elif self.cost_estimation == "actual_runtimes":
            self.db_connector.drop_index(index)
        self.current_indexes.remove(index)

    def _get_cost(self, query):
        if self.cost_estimation == "whatif":
            return self.db_connector.get_cost(query)
        elif self.cost_estimation == "actual_runtimes":
            runtime = self.db_connector.exec_query(query)[0]
            return runtime

    def complete_cost_estimation(self):
        self.completed = True

        for index in self.current_indexes.copy():
            self._unsimulate_or_drop_index(index)

        assert self.current_indexes == set()

    def _request_cache(self, query, indexes):
        q_i_hash = (query, frozenset(indexes))
        if q_i_hash in self.relevant_indexes_cache:
            relevant_indexes = self.relevant_indexes_cache[q_i_hash]
        else:
            relevant_indexes = self._relevant_indexes(query, indexes)
            self.relevant_indexes_cache[q_i_hash] = relevant_indexes

        # Check if query and corresponding relevant indexes in cache
        if (query, relevant_indexes) in self.cache:
            self.cache_hits += 1
            return self.cache[(query, relevant_indexes)]
        # If no cache hit request cost from database system
        else:
            cost = self._get_cost(query)
            self.cache[(query, relevant_indexes)] = cost
            return cost

    @staticmethod
    def _relevant_indexes(query, indexes):
        relevant_indexes = [
            x for x in indexes if any(c in query.columns for c in x.columns)
        ]
        return frozenset(relevant_indexes)
