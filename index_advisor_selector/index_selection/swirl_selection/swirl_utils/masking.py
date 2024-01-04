# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: masking
# @Author: Wei Zhou
# @Time: 2022/9/23 16:12



class MultiColumnIndexActionManager(ActionManager):
    def __init__(
        self, indexable_column_combinations, action_storage_consumptions, sb_version, max_index_width, reenable_indexes
    ):
        ActionManager.__init__(self, sb_version, max_index_width=max_index_width)

        self.indexable_column_combinations = indexable_column_combinations
        # This is the same as the Expdriment's object globally_index_candidates_flat attribute
        self.indexable_column_combinations_flat = [
            item for sublist in self.indexable_column_combinations for item in sublist
        ]
        self.number_of_actions = len(self.indexable_column_combinations_flat)
        self.number_of_columns = len(self.indexable_column_combinations[0])
        self.action_storage_consumptions = action_storage_consumptions

        self.indexable_columns = list(
            map(lambda one_column_combination: one_column_combination[0], self.indexable_column_combinations[0])
        )

        self.REENABLE_INDEXES = reenable_indexes

        self.column_to_idx = {}
        for idx, column in enumerate(self.indexable_column_combinations[0]):
            c = column[0]
            self.column_to_idx[c] = idx

        self.column_combination_to_idx = {}
        for idx, column_combination in enumerate(self.indexable_column_combinations_flat):
            cc = str(column_combination)
            self.column_combination_to_idx[cc] = idx

        self.candidate_dependent_map = {}
        for indexable_column_combination in self.indexable_column_combinations_flat:
            if len(indexable_column_combination) > max_index_width - 1:
                continue
            self.candidate_dependent_map[indexable_column_combination] = []

        for column_combination_idx, indexable_column_combination in enumerate(self.indexable_column_combinations_flat):
            if len(indexable_column_combination) < 2:
                continue
            dependent_of = indexable_column_combination[:-1]
            self.candidate_dependent_map[dependent_of].append(column_combination_idx)

    def _valid_actions_based_on_last_action(self, last_action):
        last_combination = self.indexable_column_combinations_flat[last_action]
        last_combination_length = len(last_combination)

        if last_combination_length != self.MAX_INDEX_WIDTH:
            for column_combination_idx in self.candidate_dependent_map[last_combination]:
                indexable_column_combination = self.indexable_column_combinations_flat[column_combination_idx]
                possible_extended_column = indexable_column_combination[-1]

                if possible_extended_column not in self.wl_indexable_columns:
                    continue
                if indexable_column_combination in self.current_combinations:
                    continue

                self._remaining_valid_actions.append(column_combination_idx)
                self.valid_actions[column_combination_idx] = self.ALLOWED_ACTION

        # Disable now (after the last action) invalid combinations
        for column_combination_idx in copy.copy(self._remaining_valid_actions):
            indexable_column_combination = self.indexable_column_combinations_flat[column_combination_idx]
            indexable_column_combination_length = len(indexable_column_combination)
            if indexable_column_combination_length == 1:
                continue

            if indexable_column_combination_length != last_combination_length:
                continue

            if last_combination[:-1] != indexable_column_combination[:-1]:
                continue

            if column_combination_idx in self._remaining_valid_actions:
                self._remaining_valid_actions.remove(column_combination_idx)
            self.valid_actions[column_combination_idx] = self.FORBIDDEN_ACTION

        if self.REENABLE_INDEXES and last_combination_length > 1:
            last_combination_without_extension = last_combination[:-1]

            if len(last_combination_without_extension) > 1:
                # The presence of last_combination_without_extension's parent is a precondition
                last_combination_without_extension_parent = last_combination_without_extension[:-1]
                if last_combination_without_extension_parent not in self.current_combinations:
                    return

            column_combination_idx = self.column_combination_to_idx[str(last_combination_without_extension)]
            self._remaining_valid_actions.append(column_combination_idx)
            self.valid_actions[column_combination_idx] = self.ALLOWED_ACTION

            # logging.debug(f"REENABLE_INDEXES: {last_combination_without_extension} after {last_combination}")

    def _valid_actions_based_on_workload(self, workload):
        indexable_columns = workload.indexable_columns(return_sorted=False)
        indexable_columns = indexable_columns & frozenset(self.indexable_columns)
        self.wl_indexable_columns = indexable_columns

        for indexable_column in indexable_columns:
            # only single column indexes
            for column_combination_idx, indexable_column_combination in enumerate(
                self.indexable_column_combinations[0]
            ):
                if indexable_column == indexable_column_combination[0]:
                    self.valid_actions[column_combination_idx] = self.ALLOWED_ACTION
                    self._remaining_valid_actions.append(column_combination_idx)

        assert np.count_nonzero(np.array(self.valid_actions) == self.ALLOWED_ACTION) == len(
            indexable_columns
        ), "Valid actions mismatch indexable columns"



