import math
import time
import logging
import itertools

from .selection_algorithm import DEFAULT_PARAMETER_VALUES, SelectionAlgorithm

from index_advisor_selector.index_selection.heu_selection.heu_utils.index import Index, index_merge
from index_advisor_selector.index_selection.heu_selection.heu_utils.heu_com import get_utilized_indexes, indexes_by_table, mb_to_b, b_to_mb
from index_advisor_selector.index_selection.heu_selection.heu_utils.candidate_generation import candidates_per_query, \
    syntactically_relevant_indexes, syntactically_relevant_indexes_dqn_rule, syntactically_relevant_indexes_openGauss

# budget_MB: The algorithm can utilize the specified storage budget in MB.
# max_index_width: The number of columns an index can contain at maximum.
# max_runtime_minutes: The algorithm is stopped either if all seeds are evaluated or
#                      when max_runtime_minutes is exceeded. Whatever happens first.
#                      In case of the latter, the current best solution is returned.
DEFAULT_PARAMETERS = {
    "budget_MB": DEFAULT_PARAMETER_VALUES["budget_MB"],
    "max_index_width": DEFAULT_PARAMETER_VALUES["max_index_width"],
    "max_runtime_minutes": 10,
}


# This algorithm is related to the DTA Anytime algorithm employed in SQL server.
# Details of the current version of the original algorithm are not published yet.
# See the documentation for a general description:
# https://docs.microsoft.com/de-de/sql/tools/dta/dta-utility?view=sql-server-ver15
#
# Please note, that this implementation does not reflect the behavior and performance
# of the original algorithm, which might be continuously enhanced and optimized.
class AnytimeAlgorithm(SelectionAlgorithm):
    def __init__(self, database_connector, parameters=None, process=False,
                 cand_gen=None, is_utilized=None, sel_oracle=None):
        if parameters is None:
            parameters = {}
        SelectionAlgorithm.__init__(
            self, database_connector, parameters, DEFAULT_PARAMETERS,
            process, cand_gen, is_utilized, sel_oracle
        )
        self.disk_constraint = mb_to_b(self.parameters["budget_MB"])
        self.max_index_width = self.parameters["max_index_width"]
        self.max_runtime_minutes = self.parameters["max_runtime_minutes"]

        # (0804): newly added. for number.
        self.max_indexes = self.parameters["max_indexes"]
        self.constraint = self.parameters["constraint"]

    def _calculate_best_indexes(self, workload, db_conf=None, columns=None):
        """
        1. Get the utilized hypothetical indexes (DB2Advis);
        2. Merge indexes to generate more candidates;
        3. Filter indexes that exceed the budget -> seed (AutoAdmin, enumerate_naive);
        4. Enumerate the configuration based on seed (enumerate_greedy, anytime).
        :param workload:
        :return:
        """
        logging.info("Calculating best indexes Anytime")

        # (0804): newly added. for storage budget/number.
        if (self.constraint == "number" and self.max_indexes == 0) or \
                (self.constraint == "storage" and self.disk_constraint == 0):
            return list()

        # Generate syntactically relevant candidates
        # Multi-column indexes are considered from the start.
        # (0917): newly modified.
        if self.cand_gen is None or self.cand_gen == "permutation":
            candidates = candidates_per_query(
                workload,
                self.parameters["max_index_width"],
                candidate_generator=syntactically_relevant_indexes
            )

        elif self.cand_gen == "dqn_rule":
            candidates = [syntactically_relevant_indexes_dqn_rule(db_conf, [query.text], columns,
                                                                  self.parameters["max_index_width"]) for query in
                          workload.queries]

        elif self.cand_gen == "openGauss":
            candidates = [syntactically_relevant_indexes_openGauss(db_conf, [query.text], columns,
                                                                   self.parameters["max_index_width"]) for query in
                          workload.queries]

        # (0918): newly modified.
        if self.cand_gen is None or self.is_utilized:
            # Obtain best (utilized) indexes per query
            candidates, _ = get_utilized_indexes(workload, candidates, self.cost_evaluation)
        else:
            cand_set = list()
            for cand in candidates:
                cand_set.extend(cand)
            candidates = set(cand_set)

            _ = self.cost_evaluation.calculate_cost(
                workload, candidates
                , store_size=True  # newly added.
            )

        # Candidates and configurations are merged to determine further candidates.
        # Choose two indexes from the same table and merge the columns (truncated).
        self._add_merged_indexes(candidates)

        # Remove candidates that cannot meet budget requirements
        seeds = list()
        filtered_candidates = set()
        # (0804): newly added. for reproduction.
        for candidate in sorted(list(candidates)):
            # (0804): newly added. for number.
            if self.constraint == "storage":
                if candidate.estimated_size > self.disk_constraint:
                    continue
            seeds.append({candidate})
            filtered_candidates.add(candidate)

        # For reproducible results, we sort the seeds and candidates
        # seeds = sorted(seeds, key=lambda candidate: candidate)
        # filtered_candidates = set(
        #     sorted(filtered_candidates, key=lambda candidate: candidate)
        # )
        # Each seed contains a single index in the filtered candidate.
        seeds.append(set())
        candidates = filtered_candidates

        # : newly added. for process visualization.
        if self.process:
            self.step["candidates"] = candidates

        start_time = time.time()
        # (index, cost)
        best_configuration = (None, None)

        # (0804): newly added. for reproduction.
        for i, seed in enumerate(sorted(list(seeds))):
            # : newly added. for process visualization.
            if self.process:
                self.step[i] = dict()
                self.layer = 0

            logging.info(f"Seed {i + 1} from {len(seeds)}")
            candidates_copy = candidates.copy()
            candidates_copy -= seed
            current_costs = self._simulate_and_evaluate_cost(workload, seed)
            # workload, current_indexes, current_costs, candidate_indexes, number_indexes, seed_no
            indexes, costs = self.enumerate_greedy(
                workload, seed, current_costs, candidates_copy, math.inf, i
            )
            if best_configuration[0] is None or costs < best_configuration[1]:
                best_configuration = (indexes, costs)

            current_time = time.time()
            consumed_time = current_time - start_time
            if consumed_time > self.max_runtime_minutes * 60:
                logging.info(f"Stopping after {i + 1} seeds because of timing constraints.")
                break
            else:
                logging.debug(f"Current best: {best_configuration[1]} after {consumed_time}s.")

        indexes = best_configuration[0]
        return sorted(list(indexes))

    def _add_merged_indexes(self, indexes):
        index_table_dict = indexes_by_table(indexes)
        for table in index_table_dict:
            for index1, index2 in itertools.permutations(index_table_dict[table], 2):
                merged_index = index_merge(index1, index2)
                if len(merged_index.columns) > self.max_index_width:
                    new_columns = merged_index.columns[: self.max_index_width]
                    merged_index = Index(new_columns)
                if merged_index not in indexes:
                    self.cost_evaluation.estimate_size(merged_index)
                    indexes.add(merged_index)

    # based on AutoAdminAlgorithm
    def enumerate_greedy(
            self, workload, current_indexes, current_costs, candidate_indexes, number_indexes, seed_no
    ):
        assert (
                current_indexes & candidate_indexes == set()
        ), "Intersection of current and candidate indexes must be empty"
        if len(current_indexes) >= number_indexes:
            return current_indexes, current_costs

        # (index, cost)
        best_index = (None, None)

        logging.debug(f"Searching in {len(candidate_indexes)} indexes")

        # : newly added. for process visualization.
        if self.process:
            self.step[seed_no][self.layer] = list()
            self.step["selected"].append(list())

        # (0804): newly added. for reproduction.
        for index in sorted(list(candidate_indexes)):
            # (0804): newly added. for number.
            if self.constraint == "number":
                if len(current_indexes | {index}) > self.max_indexes:
                    continue
            elif self.constraint == "storage":
                # index configuration is too large
                if sum(idx.estimated_size for idx in current_indexes | {index}) > self.disk_constraint:
                    continue

            if self.sel_oracle is None:
                cost = self._simulate_and_evaluate_cost(workload, current_indexes | {index})
            # (0917): newly added.
            elif self.sel_oracle == "cost_per_sto":
                cost = self._simulate_and_evaluate_cost(workload, current_indexes | {index})
                cost = cost * b_to_mb(index.estimated_size)
            elif self.sel_oracle == "cost_pure":
                cost = self._simulate_and_evaluate_cost(workload, current_indexes | {index})
            elif self.sel_oracle == "benefit_per_sto":
                current_cost = self._simulate_and_evaluate_cost(workload, current_indexes)
                cost = self._simulate_and_evaluate_cost(workload, current_indexes | {index})
                cost = -1 * (current_cost - cost) / b_to_mb(index.estimated_size)
            elif self.sel_oracle == "benefit_pure":
                current_cost = self._simulate_and_evaluate_cost(workload, current_indexes)
                cost = self._simulate_and_evaluate_cost(workload, current_indexes | {index})
                cost = -1 * (current_cost - cost)

            # : newly added. for process visualization.
            if self.process:
                self.step[seed_no][self.layer].append({"combination": current_indexes | {index},
                                                       "candidate": index,
                                                       "oracle": cost})



            if not best_index[0] or cost < best_index[1]:
                best_index = (index, cost)

        # no improvement -> exit.
        if best_index[0] and best_index[1] < current_costs:
            current_indexes.add(best_index[0])
            candidate_indexes.remove(best_index[0])
            current_costs = best_index[1]

            logging.debug(f"Additional best index found: {best_index}")

            # : newly added. for process visualization.
            if self.process:
                self.step["selected"][seed_no].append([item["candidate"] for item in
                                                       self.step[seed_no][self.layer]].index(best_index[0]))
                self.layer += 1

            return self.enumerate_greedy(
                workload,
                current_indexes,
                current_costs,
                candidate_indexes,
                number_indexes,
                seed_no
            )
        return current_indexes, current_costs

    # copied from AutoAdminAlgorithm
    def _simulate_and_evaluate_cost(self, workload, indexes):
        cost = self.cost_evaluation.calculate_cost(workload, indexes, store_size=True)
        return round(cost, 2)

    def get_index_candidates(self, workload, db_conf=None, columns=None):
        """
        :param workload:
        :return:
        """
        if self.cand_gen is None or self.cand_gen == "permutation":
            candidates = candidates_per_query(
                workload,
                self.parameters["max_index_width"],
                candidate_generator=syntactically_relevant_indexes
            )

        elif self.cand_gen == "dqn_rule":
            candidates = [syntactically_relevant_indexes_dqn_rule(db_conf, [query.text], columns,
                                                                  self.parameters["max_index_width"]) for query in
                          workload.queries]

        elif self.cand_gen == "openGauss":
            candidates = [syntactically_relevant_indexes_openGauss(db_conf, [query.text], columns,
                                                                   self.parameters["max_index_width"]) for query in
                          workload.queries]

        candidates_total = list()
        for cand in candidates:
            candidates_total.extend(cand)
        candidates_total = set(candidates_total)

        if self.cand_gen is None or self.is_utilized:
            # Obtain best (utilized) indexes per query
            candidates, _ = get_utilized_indexes(workload, candidates, self.cost_evaluation)
        else:
            cand_set = list()
            for cand in candidates:
                cand_set.extend(cand)
            candidates = set(cand_set)

            _ = self.cost_evaluation.calculate_cost(
                workload, candidates
                , store_size=True  # newly added.
            )

        # Candidates and configurations are merged to determine further candidates.
        # Choose two indexes from the same table and merge the columns (truncated).
        self._add_merged_indexes(candidates)

        # Remove candidates that cannot meet budget requirements
        filtered_candidates = set()
        for candidate in sorted(list(candidates)):
            if self.constraint == "storage":
                if candidate.estimated_size > self.disk_constraint:
                    continue
            filtered_candidates.add(candidate)

        return candidates_total, filtered_candidates
