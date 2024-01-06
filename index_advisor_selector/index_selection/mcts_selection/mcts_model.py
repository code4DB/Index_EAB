# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: mcts_model
# @Author: Wei Zhou
# @Time: 2022/11/2 20:39

import math
import copy

import random
import numpy as np

import hashlib
import logging

from index_advisor_selector.index_selection.mcts_selection.mcts_utils.mcts_com import mb_to_b
from index_advisor_selector.index_selection.mcts_selection.mcts_utils.mcts_workload import Workload, Table, Column, Index

# MCTS scalar (lambda).
# Larger scalar will increase exploitation, smaller will increase exploration.
# SCALAR = 1 / (2 * math.sqrt(2.0))

# Larger scalar will increase exploration, smaller will increase exploitation.
LAMBDA = math.sqrt(2.0)

# (1125):
sel_oracle = None  # None, benefit_per_sto


class State:
    def __init__(self, current_index, potential_index, constraint, cardinality, storage):
        self.current_index = current_index
        self.potential_index = potential_index

        # (0805): newly added. for `storage`.
        self.constraint = constraint
        self.cardinality = cardinality
        self.storage = storage

    def next_state(self):
        if self.constraint == "number":
            # ['catalog_sales#cs_sold_date_sk,cs_ext_discount_amt'], ['date_dim#d_date_sk']
            next_index = [random.choice(self.potential_index)]
            potential_index = sorted(list(set(self.potential_index) - set(self.current_index + next_index)))
            next_state = State(self.current_index + next_index, potential_index,
                               self.constraint, self.cardinality, self.storage)
        elif self.constraint == "storage":
            next_index = [random.choice(self.potential_index)]
            potential_index = sorted(list(set(self.potential_index) - set(self.current_index + next_index)))

            # (0806): newly added. for `storage`.
            if self.constraint == "storage":
                potential_index_filter = list()
                for index in potential_index:
                    total_size = sum([ind.estimated_size for ind in self.current_index + [index]])
                    if total_size <= mb_to_b(self.storage):
                        potential_index_filter.append(index)
                potential_index = copy.deepcopy(potential_index_filter)

            next_state = State(self.current_index + next_index, potential_index,
                               self.constraint, self.cardinality, self.storage)

        return next_state

    def is_terminal(self):
        # (0806): newly added. for `storage`.
        if self.constraint == "number":
            if len(self.current_index) == self.cardinality:
                return True
            return False
        elif self.constraint == "storage":
            # total_size = sum([index.estimated_size for index in self.current_index])
            # if total_size >= mb_to_b(self.storage):
            if len(self.potential_index) == 0:
                return True
            return False

    def get_reward(self, ind_cost, no_cost):
        if no_cost == 0:
            reward = 0  # : -1?
        else:
            reward = 1.0 - (ind_cost / no_cost)
            if sel_oracle == "benefit_per_sto":
                total_size = sum([ind.estimated_size for ind in self.current_index])
                reward = reward * total_size

            # self.reward = reward
        return reward

    # def __hash__(self):
    #     return int(hashlib.md5(str(self.moves).encode("utf-8")).hexdigest(), 16)
    #
    # def __eq__(self, other):
    #     if hash(self) == hash(other):
    #         return True
    #     return False

    def __repr__(self):
        # s = "Reward: %d; Index: %s" % (self.reward, self.current_index)
        s = "Index: %s" % self.current_index
        return s


class Node:
    def __init__(self, state, parent=None):
        self.visits = 1
        self.reward = 0.0
        self.state = state
        self.children = list()
        self.parent = parent

    def add_child(self, child_state):
        child = Node(child_state, self)
        self.children.append(child)

    def update(self, reward):
        # self.reward += reward
        self.reward = reward
        self.visits += 1

    def fully_expanded(self):
        num_moves = len(self.state.potential_index) - len(self.state.current_index)
        if len(self.children) == num_moves:
            return True
        return False

    def __repr__(self):
        s = "Node; children: %d; visits: %d; reward: %f" % (len(self.children), self.visits, self.reward)
        return s


class MCTS:
    def __init__(self, args, workload, potential_index, pg_utils,
                 cost_evaluation=None, process=False):
        self.workload = workload
        self.potential_index = potential_index
        self.pg_utils = pg_utils
        self.cost_evaluation = cost_evaluation

        # (1018): newly added.
        self.min_budget = args.min_budget
        self.early_stopping = args.early_stopping

        # for reproduce (selection, expansion, simulation).
        random.seed(args.mcts_seed)
        np.random.seed(args.mcts_seed)
        self.select_policy = args.select_policy
        self.roll_num = args.roll_num
        self.best_policy = args.best_policy

        self.best_conf = list()
        self.best_reward = 0.0

        # : newly added. for process visualization.
        self.process = process
        self.step = {"selected": list()}
        self.layer = 0

        # (0818): newly added.
        self.is_trace = args.is_trace
        self.index_trace = list()
        self.measure = {"Workload Cost": list(), "Reward": list()}

    def mcts_search(self, budget, root):
        for ite in range(budget):
            # 1. expansion: path from root node to terminal or node not be expanded.
            front = self.select_expand(root)
            # 2. simulation (roll out): sample until the terminal.
            reward = self.roll_out(front.state)
            # 3. update: the node utility and frequency backward.
            self.back_update(front, reward)

            # (0818): newly added.
            if self.is_trace:
                index, best_reward = self.extract_best(root, is_final=False)
                self.index_trace.append(index)
                self.measure["Reward"].append(best_reward)
                # logging.critical(f"The best reward ({self.best_policy}) for epoch {ite + 1} is {best_reward}.")

                if ite > self.min_budget:
                    num = np.sum(np.array(self.measure["Reward"])[-self.early_stopping:]
                                 == self.measure["Reward"][-1])
                    if num == self.early_stopping:
                        break

            # logging.critical(f"The reward for epoch {ite + 1} is {reward}.")

        # 4. best: extract best policy.
        # return self.select_node(root, 0)
        return self.extract_best(root, is_final=True)

    def select_expand(self, node):
        """
        a hack to force 'exploitation' (a random prob) in a game
        where there are many options, and
        you may never/not want to fully expand first.

        two options: 1) expand_node, 2) select_node.
        :param node:
        :return:
        """
        # loop until terminal or not be explored.
        while not node.state.is_terminal():
            if len(node.children) == 0:
                return self.expand_node(node)
            elif random.uniform(0, 1) < .5:
                node = self.select_node(node, LAMBDA)
            else:
                if not node.fully_expanded():
                    return self.expand_node(node)
                else:
                    node = self.select_node(node, LAMBDA)
        return node

    def select_node(self, node, scalar, detail=False):
        """
         1) UCT: pick the action `a` that maximizes the upper confidence bound (UCB) score;
         2) epsilon-greedy: pick an action `a` with probability that
                        is proportional to its estimated action value.
        :param node:
        :param scalar:
        :param detail:
        :return: 
        """
        if self.select_policy == "UCT" or scalar == 0:
            bestscore = -0.1
            bestchildren = list()

            if detail:
                self.step[self.layer] = list()
            for c in node.children:
                # current this uses the most vanilla MCTS formula,
                # it is worth experimenting with THRESHOLD ASCENT (TAGS).
                # exploit = c.reward / c.visits
                # explore = math.sqrt(2.0 * math.log(node.visits) / float(c.visits))
                # explore: Visits(total) / Visits(child)

                exploit = c.reward
                explore = math.sqrt(math.log(node.visits) / float(c.visits))
                score = exploit + scalar * explore

                if detail:
                    if c.parent is not None:
                        self.step[self.layer].append({"combination": c.state.current_index,
                                                      "candidate": set(c.state.current_index) -
                                                                   set(c.parent.state.current_index),
                                                      "oracle": score})
                    else:
                        self.step[self.layer].append({"combination": c.state.current_index,
                                                      "candidate": c.state.current_index,
                                                      "oracle": score})

                if score == bestscore:
                    bestchildren.append(c)
                if score > bestscore:
                    bestchildren = [c]
                    bestscore = score
            # if len(bestchildren) == 0:
            #     logger.warning("OOPS: no best child found, probably fatal.")

            # argmax, for multiple bestchildren with the same bestscore

            selected = random.choice(bestchildren)
            if detail:
                self.step["selected"].append(selected)  # selected.state.current_index
                self.layer += 1

            return selected
        elif self.select_policy == "EPSILON":
            # https://github.com/mangwang/PythonForFun/blob/master/rouletteWheelSelection.py
            rewards = [c.reward for c in node.children]
            sum_reward = random.uniform(0, np.sum(rewards))

            accum_reward = 0
            for reward, child in zip(rewards, node.children):
                accum_reward += reward
                if accum_reward >= sum_reward:
                    return child

    def expand_node(self, node):
        """
        random select from the potential index set.
        :param node:
        :return:
        """
        tried_children = [c.state for c in node.children]
        new_state = node.state.next_state()
        while new_state in tried_children and not new_state.is_terminal():
            new_state = node.state.next_state()
        node.add_child(new_state)
        return node.children[-1]

    def roll_out(self, state):
        """
        return the reward(utility) until the terminal or node not be expanded.
        :param state:
        :return:
        """
        rewards, indexes = list(), list()
        for _ in range(self.roll_num):
            while not state.is_terminal():
                state = state.next_state()

            # no_cost, ind_cost = 0, 0
            # for query in self.workload:
            #     query = query.text
            #     no_cost += self.pg_utils.get_ind_cost(query, "", mode="hypo")
            #     ind_cost += self.pg_utils.get_ind_cost(query, state.current_index, mode="hypo")

            no_cost = self.cost_evaluation.calculate_cost(Workload(self.workload), indexes=[])

            # indexes = list()
            # for index in state.current_index:
            #     tbl, col = index.split("#")
            #     col = [Column(c, Table(tbl)) for c in col.split(",")]
            #     indexes.append(Index(col))
            # ind_cost = self.cost_evaluation.calculate_cost(Workload(self.workload), indexes=indexes)

            ind_cost = self.cost_evaluation.calculate_cost(Workload(self.workload), indexes=state.current_index)

            rewards.append(state.get_reward(ind_cost, no_cost))
            indexes.append(state.current_index)

        if np.max(rewards) > self.best_reward:
            self.best_reward = np.max(rewards)
            self.best_conf = indexes[np.argmax(rewards)]

        return np.max(rewards)

    def back_update(self, node, reward):
        while node is not None:
            node.visits += 1
            if reward > node.reward:
                node.reward = reward
            # if reward > self.best_reward:
            #     # if len(node.state.current_index) == node.state.cardinality:
            #     #     self.best_reward = reward
            #     #     self.best_conf = node.state.current_index
            #     self.best_reward = reward
            #     self.best_conf = node.state.current_index
            node = node.parent

    def extract_best(self, node, is_final=False):
        """
        1) Best Configuration Explored (BCE):
            return the best configuration found during MCTS,
            among the configurations explored;
        2) Best Greedy (BG):
            use a greedy strategy to traverse the search tree.
        :return:
        """
        if self.best_policy == "BCE":
            self.step["selected"] = self.best_conf
            return self.best_conf, self.best_reward
        elif self.best_policy == "BG":
            while not node.state.is_terminal() and len(node.children) > 0:
                node = self.select_node(node, 0, is_final & self.process)

            # no_cost, ind_cost = 0, 0
            # for query in self.workload:
            #     query = query.text
            #     no_cost += self.pg_utils.get_ind_cost(query, "", mode="hypo")
            #     ind_cost += self.pg_utils.get_ind_cost(query, node.state.current_index, mode="hypo")

            no_cost = self.cost_evaluation.calculate_cost(Workload(self.workload), indexes=[])

            # indexes = list()
            # for index in node.state.current_index:
            #     tbl, col = index.split("#")
            #     col = [Column(c, Table(tbl)) for c in col.split(",")]
            #     indexes.append(Index(col))
            # ind_cost = self.cost_evaluation.calculate_cost(Workload(self.workload), indexes=indexes)

            ind_cost = self.cost_evaluation.calculate_cost(Workload(self.workload), indexes=node.state.current_index)
            self.measure["Workload Cost"].append(ind_cost)

            return node.state.current_index, node.state.get_reward(ind_cost, no_cost)
