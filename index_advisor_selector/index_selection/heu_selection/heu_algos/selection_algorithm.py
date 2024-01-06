import time
import logging

from index_advisor_selector.index_selection.heu_selection.heu_utils.cost_evaluation import CostEvaluation

# If not specified by the user,
# algorithms should use these default parameter values to
# avoid diverging values for different algorithms.
DEFAULT_PARAMETER_VALUES = {
    "budget_MB": 500,
    "max_indexes": 15,
    "max_index_width": 2,
}


class SelectionAlgorithm:
    def __init__(self, database_connector, parameters, default_parameters=None,
                 process=False, cand_gen=None, is_utilized=None, sel_oracle=None):
        if default_parameters is None:
            default_parameters = {}
        logging.debug("Init selection algorithm")
        self.did_run = False

        self.parameters = parameters
        # Store default values for missing parameters
        for key, value in default_parameters.items():
            if key not in self.parameters:
                self.parameters[key] = value

        self.database_connector = database_connector
        self.database_connector.drop_indexes()
        self.cost_evaluation = CostEvaluation(database_connector)
        if "cost_estimation" in self.parameters:
            estimation = self.parameters["cost_estimation"]
            self.cost_evaluation.cost_estimation = estimation

        # : newly added. for process visualization.
        self.process = process
        self.step = {"selected": list()}
        self.layer = 0

        # (0917): newly added.
        self.cand_gen = cand_gen
        self.is_utilized = is_utilized
        self.sel_oracle = sel_oracle

    def calculate_best_indexes(self, workload, overhead=False, db_conf=None, columns=None):
        assert self.did_run is False, "Selection algorithm can only run once."
        self.did_run = True

        estimation_num_bef = self.database_connector.cost_estimations
        estimation_duration_bef = self.database_connector.cost_estimation_duration

        simulation_num_bef = self.database_connector.simulated_indexes
        simulation_duration_bef = self.database_connector.index_simulation_duration

        time_start = time.time()
        indexes = self._calculate_best_indexes(workload, db_conf=db_conf, columns=columns)
        time_end = time.time()

        estimation_duration_aft = self.database_connector.cost_estimation_duration
        estimation_num_aft = self.database_connector.cost_estimations

        simulation_num_aft = self.database_connector.simulated_indexes
        simulation_duration_aft = self.database_connector.index_simulation_duration

        self._log_cache_hits()
        # : newly added for `swirl`.
        # self.final_cost_proportion = self._calculate_final_cost_proportion(
        #     workload, indexes
        # )

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
            return indexes

    def _calculate_best_indexes(self, workload):
        raise NotImplementedError("_calculate_best_indexes(self, " "workload) missing")

    def _log_cache_hits(self):
        hits = self.cost_evaluation.cache_hits
        requests = self.cost_evaluation.cost_requests
        logging.debug(f"Total cost cache hits:\t{hits}")
        logging.debug(f"Total cost requests:\t\t{requests}")
        if requests == 0:
            return
        ratio = round(hits * 100 / requests, 2)
        logging.debug(f"Cost cache hit ratio:\t{ratio}%")

    def _calculate_final_cost_proportion(self, workload, indexes):
        start_cost = self.cost_evaluation.calculate_cost(workload, [])
        final_cost = self.cost_evaluation.calculate_cost(workload, indexes)

        index_combination_size = 0
        for index in indexes:
            index_combination_size += index.estimated_size

        logging.info(
            (
                f"Initial cost: {start_cost:,.2f}, now: {final_cost:,.2f} "
                f"({final_cost / start_cost:.2f}) {indexes} "
                f"{index_combination_size / 1000 / 1000:.2f} MB "
                f"(of {self.parameters['budget_MB']} MB) for workload\n{workload}"
            )
        )

        return round(final_cost / start_cost * 100, 2)

    def calculate_final_cost_proportion_no_size(self, workload, indexes):
        start_cost = self.cost_evaluation.calculate_cost(workload, [])
        final_cost = self.cost_evaluation.calculate_cost(workload, indexes)

        logging.info(
            (
                f"Initial cost: {start_cost:,.2f}, now: {final_cost:,.2f} "
                f"({final_cost / start_cost:.2f}) {indexes} "
            )
        )

        return round(final_cost / start_cost * 100, 2)


class NoIndexAlgorithm(SelectionAlgorithm):
    def __init__(self, database_connector, parameters=None):
        if parameters is None:
            parameters = {}
        SelectionAlgorithm.__init__(self, database_connector, parameters)

    def _calculate_best_indexes(self, workload):
        return []


class AllIndexesAlgorithm(SelectionAlgorithm):
    def __init__(self, database_connector, parameters=None):
        if parameters is None:
            parameters = {}
        SelectionAlgorithm.__init__(self, database_connector, parameters)

    # Returns single column index for each indexable column
    def _calculate_best_indexes(self, workload):
        return workload.potential_indexes()
