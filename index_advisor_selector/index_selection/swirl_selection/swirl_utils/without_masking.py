# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: without_masking
# @Author: Wei Zhou
# @Time: 2022/9/23 16:12


class MultiColumnIndexActionManagerNonMasking(ActionManager):
    def __init__(
            self, indexable_column_combinations, action_storage_consumptions, sb_version, max_index_width,
            reenable_indexes
    ):
        ActionManager.__init__(self, sb_version, max_index_width=max_index_width)

        self.indexable_column_combinations = indexable_column_combinations
        # This is the same as the Expdriment's object globally_indexable_columns_flat attribute
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

    def update_valid_actions(self, last_action, budget, current_storage_consumption):
        assert self.indexable_column_combinations_flat[last_action] not in self.current_combinations

        last_action_column_combination = self.indexable_column_combinations_flat[last_action]

        for idx, column in enumerate(last_action_column_combination):
            status_value = 1 / (idx + 1)
            last_action_columns_idx = self.column_to_idx[column]
            self.current_action_status[last_action_columns_idx] += status_value

        self.current_combinations.add(self.indexable_column_combinations_flat[last_action])

        self._valid_actions_based_on_last_action(last_action)
        self._valid_actions_based_on_budget(budget, current_storage_consumption)

        return np.array(self.valid_actions), True

    def _valid_actions_based_on_last_action(self, last_action):
        pass

    def _valid_actions_based_on_workload(self, workload):
        self.valid_actions = [self.ALLOWED_ACTION for action in range(self.number_of_actions)]

    def _valid_actions_based_on_budget(self, budget, current_storage_consumption):
        pass
