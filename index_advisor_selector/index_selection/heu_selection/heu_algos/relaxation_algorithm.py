import itertools
import logging

from .selection_algorithm import DEFAULT_PARAMETER_VALUES, SelectionAlgorithm

from index_advisor_selector.index_selection.heu_selection.heu_utils.index import Index, index_merge, index_split
from index_advisor_selector.index_selection.heu_selection.heu_utils.heu_com import get_utilized_indexes, indexes_by_table, mb_to_b, b_to_mb
from index_advisor_selector.index_selection.heu_selection.heu_utils.candidate_generation import candidates_per_query, \
    syntactically_relevant_indexes, syntactically_relevant_indexes_dqn_rule, syntactically_relevant_indexes_openGauss

# allowed_transformations: The algorithm transforms index configurations. Via this
#                          parameter, the allowed transformations can be chosen.
#                          In the original paper, 5 transformations are documented.
#                          Except for "Promotion to clustered" all transformations are
#                          implemented and part of the algorithm's default configuration.
# budget_MB: The algorithm can utilize the specified storage budget in MB.
# max_index_width: The number of columns an index can contain at maximum.
DEFAULT_PARAMETERS = {
    "allowed_transformations": ["splitting", "merging", "prefixing", "removal"],
    "budget_MB": DEFAULT_PARAMETER_VALUES["budget_MB"],
    "max_index_width": DEFAULT_PARAMETER_VALUES["max_index_width"],
}


# This algorithm is a reimplementation of Bruno's and Chaudhuri's relaxation-based
# approach to physical database design.
# Details can be found in the original paper:
# Nicolas Bruno, Surajit Chaudhuri: Automatic Physical Database Tuning:
# A Relaxation-based Approach. SIGMOD Conference 2005: 227-238
class RelaxationAlgorithm(SelectionAlgorithm):
    def __init__(self, database_connector, parameters=None, process=False,
                 cand_gen=None, is_utilized=None, sel_oracle=None):
        if parameters is None:
            parameters = {}
        SelectionAlgorithm.__init__(
            self, database_connector, parameters, DEFAULT_PARAMETERS,
            process, cand_gen, is_utilized, sel_oracle
        )
        self.disk_constraint = mb_to_b(self.parameters["budget_MB"])
        self.transformations = self.parameters["allowed_transformations"]
        self.max_index_width = self.parameters["max_index_width"]
        assert set(self.transformations) <= {
            "splitting",
            "merging",
            "prefixing",
            "removal",
        }

        # (0804): newly added. for number.
        self.max_indexes = self.parameters["max_indexes"]
        self.constraint = self.parameters["constraint"]

    def _calculate_best_indexes(self, workload, db_conf=None, columns=None):
        """
        1. Generate all syntactically relevant indexes;
        2. Get the initial configuration set including all utilized hypothetical indexes (same as DB2Advis);
        3. Relax the configuration set based on 4 transformations(prefixing, removal, merging, splitting).
           Strategy: pick one of the best (min(relaxed penalty)) relaxed transformations each time.
           1) relaxed_storage_savings <= 0, continue,
              relaxed_considered_storage_savings = min(relaxed_storage_savings, cp_size - self.disk_constraint);
           2) relaxed penalty: relaxed_cost_increase (< 0) * relaxed_storage_savings;
                               relaxed_cost_increase (> 0) / relaxed_considered_storage_savings.

        :param workload:
        :return:
        """
        logging.info("Calculating best indexes Relaxation")

        # (0804): newly added. for storage budget/number.
        if (self.constraint == "number" and self.max_indexes == 0) or \
                (self.constraint == "storage" and self.disk_constraint == 0):
            return list()

        # Generate syntactically relevant candidates
        # (0917): newly added.
        if self.cand_gen is None or self.cand_gen == "permutation":
            candidates = candidates_per_query(
                workload,
                self.parameters["max_index_width"],
                candidate_generator=syntactically_relevant_indexes,
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
            # Obtain the utilized indexes considering every single query
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

        # : newly added. for process visualization.
        if self.process:
            self.step["candidates"] = candidates.copy()

        # CP in Figure 5
        # (0804): newly added. for reproduction.
        cp = sorted(list(candidates.copy()))
        cp_size = sum(index.estimated_size for index in cp)
        cp_cost = self.cost_evaluation.calculate_cost(workload, cp, store_size=True)
        while True:
            # (0804): newly added. for storage budget/number.
            if self.constraint == "storage" and cp_size <= self.disk_constraint:
                break
            if self.constraint == "number" and len(cp) <= self.max_indexes:
                break

            logging.debug(
                f"Size of current configuration: {cp_size}. "
                f"Budget: {self.disk_constraint}."
            )

            # Pick a configuration that can be relaxed
            # : Currently only one is considered

            # Relax the configuration
            best_relaxed = None
            best_relaxed_size = None
            lowest_relaxed_penalty = None

            cp_by_table = indexes_by_table(cp)

            # : newly added. for process visualization.
            if self.process:
                self.step[self.layer] = list()

            for transformation in self.transformations:
                for (relaxed, relaxed_storage_savings) \
                        in self._configurations_by_transformation(cp, cp_by_table, transformation):
                    relaxed_cost = self.cost_evaluation.calculate_cost(workload, relaxed, store_size=True)
                    # Note, some transformations could also decrease the cost,
                    # indicated by a negative value.
                    relaxed_cost_increase = relaxed_cost - cp_cost

                    # # (0804): newly added. for storage budget/number.
                    # if self.constraint == "storage":

                    # Some transformations could increase or not affect the storage consumption.
                    # For termination of the algorithm, the storage savings must be positive.
                    if relaxed_storage_savings <= 0:
                        continue
                    relaxed_considered_storage_savings = min(relaxed_storage_savings,
                                                             cp_size - self.disk_constraint)

                    if relaxed_cost_increase < 0:
                        # For a (fixed) cost decrease,
                        # (indicated by a negative value for relaxed_cost_increase),
                        # higher storage savings produce a lower penalty.
                        # (0917): newly added.
                        if self.sel_oracle is None:
                            relaxed_penalty = relaxed_cost_increase * relaxed_storage_savings
                        elif self.sel_oracle == "cost_per_sto":
                            relaxed_penalty = relaxed_cost * b_to_mb(sum([index.estimated_size for index in relaxed]))
                        elif self.sel_oracle == "cost_pure":
                            relaxed_penalty = relaxed_cost
                        elif self.sel_oracle == "benefit_per_sto":
                            relaxed_penalty = relaxed_cost_increase * relaxed_storage_savings
                        elif self.sel_oracle == "benefit_pure":
                            relaxed_penalty = relaxed_cost_increase
                    else:
                        # (0917): newly added.
                        if self.sel_oracle is None:
                            relaxed_penalty = (relaxed_cost_increase / relaxed_considered_storage_savings)
                        elif self.sel_oracle == "cost_per_sto":
                            relaxed_penalty = relaxed_cost * b_to_mb(sum([index.estimated_size for index in relaxed]))
                        elif self.sel_oracle == "cost_pure":
                            relaxed_penalty = relaxed_cost
                        elif self.sel_oracle == "benefit_per_sto":
                            relaxed_penalty = (relaxed_cost_increase / relaxed_considered_storage_savings)
                        elif self.sel_oracle == "benefit_pure":
                            relaxed_penalty = relaxed_cost_increase

                    # elif self.constraint == "number":
                    #     relaxed_penalty = relaxed_cost_increase

                    # : newly added. for process visualization.
                    if self.process:
                        self.step[self.layer].append({"combination": relaxed,
                                                      "candidate": relaxed,
                                                      "oracle": relaxed_penalty})

                    if best_relaxed is None or relaxed_penalty < lowest_relaxed_penalty:
                        # set new best relaxed configuration
                        best_relaxed = relaxed
                        # (0804): newly added. for storage budget/number.
                        if self.constraint == "storage":
                            best_relaxed_size = cp_size - relaxed_considered_storage_savings
                        lowest_relaxed_penalty = relaxed_penalty

            cp = best_relaxed
            # (0804): newly added. for storage budget/number.
            if self.constraint == "storage":
                cp_size = best_relaxed_size

            # : newly added. for process visualization.
            if self.process:
                self.step["selected"].append(
                    [item["oracle"] for item in self.step[self.layer]].index(lowest_relaxed_penalty))
                self.layer += 1

        # (0804): newly added. for reproduction.
        return sorted(list(cp))

    def _configurations_by_transformation(
            self, input_configuration, input_configuration_by_table, transformation
    ):
        # 1) relaxation by adding the prefix and removing the index.
        if transformation == "prefixing":
            # (0804): newly added. for reproduction.
            for index in sorted(list(input_configuration)):
                for prefix in index.prefixes():
                    relaxed = set(input_configuration.copy())
                    relaxed.remove(index)
                    relaxed_storage_savings = index.estimated_size
                    if prefix not in relaxed:
                        relaxed.add(prefix)
                        self.cost_evaluation.estimate_size(prefix)
                        relaxed_storage_savings -= prefix.estimated_size
                    yield relaxed, relaxed_storage_savings
        # 2) relaxation by removing the index directly.
        elif transformation == "removal":
            # (0804): newly added. for reproduction.
            for index in sorted(list(input_configuration)):
                relaxed = set(input_configuration.copy())
                relaxed.remove(index)
                yield relaxed, index.estimated_size
        # 3) relaxation by merging two index and removing them.
        elif transformation == "merging":  # same as anytime
            # (0804): newly added. for reproduction.
            for table in sorted(list(input_configuration_by_table)):
                for index1, index2 in itertools.permutations(
                        input_configuration_by_table[table], 2
                ):
                    relaxed = set(input_configuration.copy())
                    merged_index = index_merge(index1, index2)
                    if len(merged_index.columns) > self.max_index_width:
                        new_columns = merged_index.columns[: self.max_index_width]
                        merged_index = Index(new_columns)

                    relaxed -= {index1, index2}
                    relaxed_storage_savings = (
                            index1.estimated_size + index2.estimated_size
                    )
                    if merged_index not in relaxed:
                        relaxed.add(merged_index)
                        self.cost_evaluation.estimate_size(merged_index)
                        relaxed_storage_savings -= merged_index.estimated_size
                    yield relaxed, relaxed_storage_savings
        # 4) relaxation by splitting two index into three indexes: (common, residual_1, residual_2).
        elif transformation == "splitting":
            # (0804): newly added. for reproduction.
            for table in sorted(list(input_configuration_by_table)):
                for index1, index2 in itertools.permutations(
                        input_configuration_by_table[table], 2
                ):
                    relaxed = set(input_configuration.copy())
                    indexes_by_splitting = index_split(index1, index2)
                    if indexes_by_splitting is None:
                        # no splitting for index permutation possible
                        continue
                    relaxed -= {index1, index2}
                    relaxed_storage_savings = (
                            index1.estimated_size + index2.estimated_size
                    )
                    for index in indexes_by_splitting - relaxed:
                        relaxed.add(index)
                        self.cost_evaluation.estimate_size(index)
                        relaxed_storage_savings -= index.estimated_size
                    yield relaxed, relaxed_storage_savings

    def get_index_candidates(self, workload, db_conf=None, columns=None):
        """
        :param workload:
        :return:
        """
        # Generate syntactically relevant candidates
        if self.cand_gen is None or self.cand_gen == "permutation":
            candidates = candidates_per_query(
                workload,
                self.parameters["max_index_width"],
                candidate_generator=syntactically_relevant_indexes,
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

        # Obtain the utilized indexes considering every single query
        utilized_indexes, _ = get_utilized_indexes(workload, candidates, self.cost_evaluation)

        return candidates_total, utilized_indexes
