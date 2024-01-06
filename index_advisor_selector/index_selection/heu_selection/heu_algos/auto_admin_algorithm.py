import logging
import itertools

from .selection_algorithm import DEFAULT_PARAMETER_VALUES, SelectionAlgorithm

from index_advisor_selector.index_selection.heu_selection.heu_utils.index import Index
from index_advisor_selector.index_selection.heu_selection.heu_utils.workload import Workload
from index_advisor_selector.index_selection.heu_selection.heu_utils.heu_com import mb_to_b, b_to_mb

# max_indexes: The algorithm stops as soon as it has selected #max_indexes indexes
# max_indexes_naive: Number of indexes selected by a naive enumeration, see
#                    enumerate_naive() for further details.
# max_index_width: The number of columns an index can contain at maximum.
DEFAULT_PARAMETERS = {
    "max_indexes": DEFAULT_PARAMETER_VALUES["max_indexes"],
    "max_indexes_naive": 2,
    "max_index_width": DEFAULT_PARAMETER_VALUES["max_index_width"],
}


# This algorithm resembles the index selection algorithm published in 1997 by Chaudhuri
# and Narasayya. Details can be found in the original paper:
# Surajit Chaudhuri, Vivek R. Narasayya: An Efficient Cost-Driven Index Selection
# Tool for Microsoft SQL Server. VLDB 1997: 146-155
#
# Please note, that this implementation does not reflect the behavior and performance
# of the original algorithm, which might be continuously enhanced and optimized.
class AutoAdminAlgorithm(SelectionAlgorithm):
    def __init__(self, database_connector, parameters, process=False,
                 cand_gen=None, is_utilized=None, sel_oracle=None):
        SelectionAlgorithm.__init__(
            self, database_connector, parameters, DEFAULT_PARAMETERS,
            process, cand_gen, is_utilized, sel_oracle
        )
        self.max_indexes = self.parameters["max_indexes"]
        self.max_indexes_naive = min(
            self.parameters["max_indexes_naive"], self.max_indexes
        )
        self.max_index_width = self.parameters["max_index_width"]

        # (0804): newly added. for storage budget.
        self.budget = mb_to_b(self.parameters["budget_MB"])
        self.constraint = self.parameters["constraint"]

        if self.constraint == "storage":
            self.max_indexes = 100

    def _calculate_best_indexes(self, workload, db_conf=None, columns=None):
        """
        1. Candidate selection based on every single query (same enumeration process);
        2. Strategy: enumerate_naive(brute-force) + enumerate_greedy;
        3. Increment the index width by 1 to generate multi-column indexes.
        :param workload:
        :return:
        """
        logging.info("Calculating best indexes AutoAdmin")

        # (0804): newly added. for storage budget/number.
        if (self.constraint == "number" and self.max_indexes == 0) or \
                (self.constraint == "storage" and self.budget == 0):
            return list()

        # : newly added. for process visualization.
        if self.process:
            self.step["candidates"] = dict()

        # Set potential indexes for first iteration: single-column index
        potential_indexes = workload.potential_indexes()
        for current_max_index_width in range(1, self.max_index_width + 1):
            candidates = self.select_index_candidates(workload, potential_indexes)
            # : newly added. for process visualization.
            if self.process:
                self.layer = 0
                self.step[current_max_index_width - 1] = dict()
                self.step["candidates"][current_max_index_width] = candidates

            indexes = self.enumerate_combinations(workload, candidates,
                                                  is_seed=False, iter=current_max_index_width - 1)
            assert indexes <= candidates, "Indexes must be a subset of candidate indexes"

            if current_max_index_width < self.max_index_width:
                # Update potential indexes for the next iteration
                potential_indexes = indexes | self.create_multicolumn_indexes(
                    workload, indexes
                )
        return indexes

    def select_index_candidates(self, workload, potential_indexes):
        candidates = set()
        for query in workload.queries:
            logging.debug(f"Find candidates for query\t{query}...")
            # Create a workload consisting of one query
            query_workload = Workload([query])
            indexes = self._potential_indexes_for_query(query, potential_indexes)
            # : newly added. for process visualization.
            candidates |= self.enumerate_combinations(query_workload, indexes, is_seed=True)

        # (0804): newly added. for reproduction.
        candidates = set(sorted(list(candidates)))
        logging.info(
            f"Number of candidates: {len(candidates)}\n" f"Candidates: {candidates}"
        )
        return candidates

    def _potential_indexes_for_query(self, query, potential_indexes):
        indexes = set()
        # (0804): newly added. for reproduction.
        for index in sorted(list(potential_indexes)):
            # The leading index column must be referenced by the query
            if index.columns[0] in query.columns:
                indexes.add(index)

        return indexes

    def enumerate_combinations(self, workload, candidate_indexes, is_seed=True, iter=None):
        log_out = (
            f"Start Enumeration\n"
            f"\tNumber of candidate indexes: {len(candidate_indexes)}\n"
            f"\tNumber of indexes to be selected: {self.max_indexes}"
        )
        logging.debug(log_out)

        number_indexes_naive = min(self.max_indexes_naive, len(candidate_indexes))
        # picks an optimal configuration of size m (where m <= k) as the “seed” (brute-force)
        current_indexes, costs = self.enumerate_naive(
            workload, candidate_indexes, number_indexes_naive, is_seed, iter
        )

        log_out = (
            f"lowest cost (naive): {costs}\n"
            f"\tlowest cost indexes (naive): {current_indexes}"
        )
        logging.debug(log_out)

        # : len(candidate_indexes)? current indexes all chosen from the candidate indexes.
        number_indexes = min(self.max_indexes, len(candidate_indexes))
        indexes, costs = self.enumerate_greedy(
            workload,
            current_indexes,
            costs,
            # (0804): len(current_indexes) > min(self.max_indexes, len(candidate_indexes))?
            candidate_indexes - current_indexes,
            number_indexes,
            is_seed,
            iter
        )

        log_out = (
            f"lowest cost (greedy): {costs}\n"
            f"\tlowest cost indexes (greedy): {indexes}\n"
            f"(greedy): number indexes {len(indexes)}\n"
        )
        logging.debug(log_out)

        return set(indexes)

    def enumerate_naive(self, workload, candidate_indexes, number_indexes_naive, is_seed=True, iter=None):
        """
        1. Brute-Force: picks an optimal configuration of size m (where m <= k, at most `number_indexes_naive`) as the “seed”.
        :param workload:
        :param candidate_indexes:
        :param number_indexes_naive:
        :return:
        """
        lowest_cost_indexes = set()
        # : initialized by `self.cost_evaluation.calculate_cost(self.workload, indexes=lowest_cost_indexes)`
        lowest_cost = None

        # : newly added. for process visualization.
        if self.process and not is_seed:
            self.step[iter][self.layer] = list()
            self.step["selected"].append(list())

        for number_of_indexes in range(1, number_indexes_naive + 1):
            # (0804): newly added. for reproduction.
            for index_combination in sorted(list(itertools.combinations(
                    candidate_indexes, number_of_indexes))):

                if self.sel_oracle is None:
                    cost = self._simulate_and_evaluate_cost(workload, index_combination)
                elif self.sel_oracle == "cost_per_sto":
                    cost = self._simulate_and_evaluate_cost(workload, index_combination)
                    cost = cost * b_to_mb(sum(index.estimated_size for index in index_combination))
                elif self.sel_oracle == "cost_pure":
                    cost = self._simulate_and_evaluate_cost(workload, index_combination)
                elif self.sel_oracle == "benefit_per_sto":
                    current_cost = self._simulate_and_evaluate_cost(workload, set())
                    cost = self._simulate_and_evaluate_cost(workload, index_combination)
                    cost = -1 * (current_cost - cost) / b_to_mb(sum(index.estimated_size for index in index_combination))
                elif self.sel_oracle == "benefit_pure":
                    current_cost = self._simulate_and_evaluate_cost(workload, set())
                    cost = self._simulate_and_evaluate_cost(workload, index_combination)
                    cost = -1 * (current_cost - cost)

                if self.constraint == "storage":
                    total_size = sum(index.estimated_size for index in index_combination)
                    if total_size > self.budget:
                        continue

                # : newly added. for process visualization.
                if self.process and not is_seed:
                    self.step[iter][self.layer].append({"combination": index_combination,
                                                        "candidate": index_combination,
                                                        "oracle": cost})

                if not lowest_cost or cost < lowest_cost:
                    lowest_cost_indexes = index_combination
                    lowest_cost = cost

        # : newly added. for process visualization.
        # (0416): newly added. `lowest_cost is not None`,
        #  `ValueError: None is not in list`.
        if self.process and not is_seed and lowest_cost is not None:
            # self.step["selected"][iter].append(np.argmin([item["oracle"] for item in self.step[iter][self.layer]]))
            self.step["selected"][iter].append(
                [item["oracle"] for item in self.step[iter][self.layer]].index(lowest_cost))
            self.layer += 1

        if not lowest_cost:
            lowest_cost = self._simulate_and_evaluate_cost(workload, set())

        return set(lowest_cost_indexes), lowest_cost

    def enumerate_greedy(
            self, workload, current_indexes, current_costs, candidate_indexes, number_indexes, is_seed=True, iter=None
    ):
        """
        2. Greedy: min(cost(S U I)), for I != I'
        :param workload:
        :param current_indexes:
        :param current_costs:
        :param candidate_indexes:
        :param number_indexes:
        :param is_seed:
        :param iter:
        :return:
        """
        assert (
                current_indexes & candidate_indexes == set()
        ), "Intersection of current and candidate indexes must be empty"
        if self.constraint == "number":
            # : ? current indexes all chosen from the candidate indexes.
            if len(current_indexes) >= number_indexes:
                return current_indexes, current_costs
        # (0804): newly added. for storage budget.
        elif self.constraint == "storage":
            total_size = sum(index.estimated_size for index in current_indexes)
            if total_size > self.budget:
                return current_indexes, current_costs

        # (index, cost)
        best_index = (None, None)

        logging.debug(f"Searching in {len(candidate_indexes)} indexes")

        # : newly added. for process visualization.
        if self.process and not is_seed:
            self.step[iter][self.layer] = list()

        # (0804): newly added. for reproduction.
        for index in sorted(list(candidate_indexes)):
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

            if self.constraint == "storage":
                index_combination = current_indexes | {index}
                total_size = sum(index.estimated_size for index in index_combination)
                if total_size > self.budget:
                    continue

            # : newly added. for process visualization.
            if self.process and not is_seed:
                self.step[iter][self.layer].append({"combination": current_indexes | {index},
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
            if self.process and not is_seed:
                self.step["selected"][iter].append(
                    [item["candidate"] for item in self.step[iter][self.layer]].index(best_index[0]))
                self.layer += 1

            return self.enumerate_greedy(
                workload,
                current_indexes,
                current_costs,
                candidate_indexes,
                number_indexes,
                is_seed,
                iter
            )

        # (0804): newly added. for reproduction.
        current_indexes = set(sorted(list(current_indexes)))
        return current_indexes, current_costs

    def _simulate_and_evaluate_cost(self, workload, indexes):
        cost = self.cost_evaluation.calculate_cost(workload, indexes, store_size=True)
        return round(cost, 2)

    def create_multicolumn_indexes(self, workload, indexes):
        multicolumn_candidates = set()
        for index in indexes:
            # Extend the index with all indexable columns of the same table,
            # that are not already part of the index
            for column in (set(index.table().columns) & set(workload.indexable_columns())) - set(index.columns):
                multicolumn_candidates.add(Index(index.columns + (column,)))
        return multicolumn_candidates
