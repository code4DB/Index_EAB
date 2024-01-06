import logging
import operator
from abc import abstractmethod

import constants


class BaseOracle:
    def __init__(self, constraint, max_memory=500, max_count=5, process=False):
        self.constraint = constraint
        self.max_memory = max_memory
        self.max_count = max_count

        # (0818): newly added.
        self.process = process
        self.step = {"selected": list()}
        self.layer = 0

    @abstractmethod
    def get_super_arm(self, upper_bounds, context_vectors, bandit_arms):
        pass

    @staticmethod
    def arm_satisfy_predicate(arm, predicate, table_name):
        """
        Check if the bandit arm can be helpful for a given predicate
        :param arm: bandit arm
        :param predicate: predicate that we wanna test against
        :param table_name: table name
        :return: boolean
        """
        if table_name == arm.table_name and predicate == arm.index_cols[0]:
            return True
        return False

    @staticmethod
    def removed_covered_budget_v2(arm_ucb_dict, chosen_id, bandit_arms, remaining_memory):
        """
        second version which is based on the remaining memory and already chosen arms

        :param arm_ucb_dict: dictionary of arms and upper confidence bounds
        :param chosen_id: chosen arm in this round
        :param bandit_arms: Bandit arm list
        :param remaining_memory: max_memory - used_memory
        :return: reduced arm list
        """
        reduced_arm_ucb_dict = {}
        for arm_id in arm_ucb_dict:
            # (0814): 1. consumed by certain chosen index (prefix); 2. exceed the storage budget.
            if not (bandit_arms[arm_id] <= bandit_arms[chosen_id] or bandit_arms[arm_id].memory > remaining_memory):
                reduced_arm_ucb_dict[arm_id] = arm_ucb_dict[arm_id]
        return reduced_arm_ucb_dict

    @staticmethod
    def removed_covered_tables(arm_ucb_dict, chosen_id, bandit_arms, table_count):
        """
        Remove the arms whose table exists `MAX_INDEXES_PER_TABLE` indexes already.
        :param arm_ucb_dict: dictionary of arms and upper confidence bounds
        :param chosen_id: chosen arm in this round
        :param bandit_arms: Bandit arm list
        :param table_count: count of indexes already chosen for each table
        :return: reduced arm list
        """
        reduced_arm_ucb_dict = {}
        for arm_id in arm_ucb_dict:
            if not (bandit_arms[arm_id].table_name == bandit_arms[chosen_id].table_name
                    and table_count[bandit_arms[arm_id].table_name] >= constants.MAX_INDEXES_PER_TABLE):
                reduced_arm_ucb_dict[arm_id] = arm_ucb_dict[arm_id]
        return reduced_arm_ucb_dict

    @staticmethod
    def removed_covered_clusters(arm_ucb_dict, chosen_id, bandit_arms):
        """
        todo(0814): cluster: table_name + '_' + str(query_id) + '_all'?
        :param arm_ucb_dict: dictionary of arms and upper confidence bounds
        :param chosen_id: chosen arm in this round
        :param bandit_arms: Bandit arm list
        :return: reduced arm list
        """
        reduced_arm_ucb_dict = {}
        for arm_id in arm_ucb_dict:
            if not (bandit_arms[arm_id].table_name == bandit_arms[chosen_id].table_name
                    and bandit_arms[chosen_id].cluster is not None
                    and bandit_arms[arm_id].cluster is not None
                    and bandit_arms[arm_id].cluster == bandit_arms[chosen_id].cluster):
                reduced_arm_ucb_dict[arm_id] = arm_ucb_dict[arm_id]
        return reduced_arm_ucb_dict

    @staticmethod
    def removed_covered_queries_v2(arm_ucb_dict, chosen_id, bandit_arms):
        """
        When covering index is selected for a query we gonna remove all other arms from that query.

        :param arm_ucb_dict: dictionary of arms and upper confidence bounds
        :param chosen_id: chosen arm in this round
        :param bandit_arms: Bandit arm list
        :return: reduced arm list
        """
        reduced_arm_ucb_dict = {}
        for arm_id in arm_ucb_dict:
            query_ids = bandit_arms[chosen_id].query_ids
            for query_id in query_ids:
                if (bandit_arms[arm_id].table_name == bandit_arms[chosen_id].table_name
                        and bandit_arms[chosen_id].is_include == 1
                        and query_id in bandit_arms[arm_id].query_ids):
                    bandit_arms[arm_id].query_ids.remove(query_id)
            if bandit_arms[arm_id].query_ids != set():
                reduced_arm_ucb_dict[arm_id] = arm_ucb_dict[arm_id]
        return reduced_arm_ucb_dict

    @staticmethod
    def removed_low_expected_rewards(arm_ucb_dict, threshold):
        """
        It make sense to remove arms with low expected reward
        :param arm_ucb_dict: dictionary of arms and upper confidence bounds
        :param threshold: expected reward threshold
        :return: reduced arm list
        """
        reduced_arm_ucb_dict = {}
        for arm_id in arm_ucb_dict:
            if arm_ucb_dict[arm_id] > threshold:
                reduced_arm_ucb_dict[arm_id] = arm_ucb_dict[arm_id]
        return reduced_arm_ucb_dict

    @staticmethod
    def removed_same_prefix(arm_ucb_dict, chosen_id, bandit_arms, prefix_length):
        """
        One index for one query for table.

        :param arm_ucb_dict: dictionary of arms and upper confidence bounds
        :param chosen_id: chosen arm in this round
        :param bandit_arms: Bandit arm list
        :param prefix_length: This is the length of the prefix
        :return: reduced arm list
        """
        reduced_arm_ucb_dict = {}
        if len(bandit_arms[chosen_id].index_cols) < prefix_length:
            return arm_ucb_dict
        else:
            for arm_id in arm_ucb_dict:
                if (bandit_arms[arm_id].table_name == bandit_arms[chosen_id].table_name and
                        len(bandit_arms[arm_id].index_cols) > prefix_length):
                    for i in range(prefix_length):
                        if bandit_arms[arm_id].index_cols[i] != bandit_arms[chosen_id].index_cols[i]:
                            reduced_arm_ucb_dict[arm_id] = arm_ucb_dict[arm_id]
                            continue
                else:
                    reduced_arm_ucb_dict[arm_id] = arm_ucb_dict[arm_id]
        return reduced_arm_ucb_dict


class OracleV7(BaseOracle):
    def get_super_arm(self, upper_bounds, context_vectors, bandit_arms, is_final=False):
        used_memory = 0
        chosen_arms = list()
        arm_ucb_dict = dict()
        table_count = dict()

        for i in range(len(bandit_arms)):
            arm_ucb_dict[i] = upper_bounds[i]

        arm_ucb_dict = self.removed_low_expected_rewards(arm_ucb_dict, threshold=0)
        while len(arm_ucb_dict) > 0:
            max_ucb_arm_id = max(arm_ucb_dict.items(), key=operator.itemgetter(1))[0]

            # (0818): newly added.
            if self.process and is_final:
                self.step[self.layer] = list()
                for cand, score in arm_ucb_dict.items():
                    self.step[self.layer].append(
                        {"combination": [bandit_arms[arm].index_cols for arm in chosen_arms + [cand]],
                         "candidate": bandit_arms[cand].index_cols,
                         "oracle": score[0]})
                self.layer += 1

            if self.constraint == "storage" and \
                    bandit_arms[max_ucb_arm_id].memory < self.max_memory - used_memory:
                chosen_arms.append(max_ucb_arm_id)

                # (0815): newly modified.
                arm_ucb_dict.pop(max_ucb_arm_id)

                used_memory += bandit_arms[max_ucb_arm_id].memory
                if bandit_arms[max_ucb_arm_id].table_name in table_count:
                    table_count[bandit_arms[max_ucb_arm_id].table_name] += 1
                else:
                    table_count[bandit_arms[max_ucb_arm_id].table_name] = 1
                # (0814): violate `MAX_INDEXES_PER_TABLE` constraint.
                arm_ucb_dict = self.removed_covered_tables(arm_ucb_dict, max_ucb_arm_id, bandit_arms, table_count)
                # (0814): cluster: table_name + '_' + str(query_id) + '_all'?
                arm_ucb_dict = self.removed_covered_clusters(arm_ucb_dict, max_ucb_arm_id, bandit_arms)
                # (0814): covering index is selected for a query.
                arm_ucb_dict = self.removed_covered_queries_v2(arm_ucb_dict, max_ucb_arm_id, bandit_arms)
                # (0814): violate the storage budget. to be modified. for `number`.
                arm_ucb_dict = self.removed_covered_budget_v2(arm_ucb_dict, max_ucb_arm_id, bandit_arms,
                                                              self.max_memory - used_memory)
                # (0814): One index for one query for table?
                arm_ucb_dict = self.removed_same_prefix(arm_ucb_dict, max_ucb_arm_id,
                                                        bandit_arms, prefix_length=1)
            # (0814): newly modified.
            elif self.constraint == "number":
                chosen_arms.append(max_ucb_arm_id)

                # (0815): newly modified.
                arm_ucb_dict.pop(max_ucb_arm_id)

                if bandit_arms[max_ucb_arm_id].table_name in table_count:
                    table_count[bandit_arms[max_ucb_arm_id].table_name] += 1
                else:
                    table_count[bandit_arms[max_ucb_arm_id].table_name] = 1
                # (0814): violate `MAX_INDEXES_PER_TABLE` constraint.
                arm_ucb_dict = self.removed_covered_tables(arm_ucb_dict, max_ucb_arm_id, bandit_arms, table_count)
                # (0814): cluster: table_name + '_' + str(query_id) + '_all'?
                arm_ucb_dict = self.removed_covered_clusters(arm_ucb_dict, max_ucb_arm_id, bandit_arms)
                # (0814): covering index is selected for a query.
                arm_ucb_dict = self.removed_covered_queries_v2(arm_ucb_dict, max_ucb_arm_id, bandit_arms)
                # (0814): One index for one query for table?
                arm_ucb_dict = self.removed_same_prefix(arm_ucb_dict, max_ucb_arm_id, bandit_arms, 1)
            else:
                arm_ucb_dict.pop(max_ucb_arm_id)

            # (0814): newly added. for `number`.
            if self.constraint == "number" and len(chosen_arms) >= self.max_count:
                break

        logging.info(f"The chosen arms/indexes ({len(chosen_arms)}): {chosen_arms}.")

        return chosen_arms
