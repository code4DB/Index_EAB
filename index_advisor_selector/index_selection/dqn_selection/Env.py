import math
import numpy as np

from dqn_utils import PostgreSQL as pg


class Env:
    def __init__(self, args, workload, frequency, candidates, mode, a):
        self.args = args
        # (0813): newly added.
        self.constraint = args.constraint
        self.max_count = 0
        self.max_storage = 0
        self.current_storage_sum = 0

        self.workload = workload
        self.candidates = candidates

        # pre-built index (determined by greedy policy)
        self.pre_create = list()
        self.index_no, self.sizes = list(), list()

        # create real/hypothetical index
        self.mode = mode
        self.pg_client1 = pg.PGHypo(args.conf_load)
        # only for `checkout()`
        self.pg_client2 = pg.PGHypo(args.conf_load)

        # [1265, 897, 643, 1190, 521, 1688, 778, 1999, 1690, 1433, 1796, 1266, 1046, 1353]
        # self._frequencies = frequency
        # self.frequencies = np.array(self._frequencies) / np.array(self._frequencies).sum()
        self.frequencies = frequency

        # state info
        self.init_cost = np.array(self.pg_client1.get_queries_cost(workload)) * self.frequencies
        self.init_cost_sum = self.init_cost.sum()

        # (0808): to be modified.
        # self.init_state = np.append(self.init_cost, np.zeros((len(candidates),), dtype=np.float))
        self.init_state = np.append(self.frequencies, np.zeros((len(candidates),), dtype=np.float))

        self.last_state = self.init_state
        self.last_cost = self.init_cost
        self.last_cost_sum = self.init_cost_sum

        # utility info
        self.current_index_count = 0
        self.current_index = np.zeros((len(candidates),), dtype=np.float)

        # monitor info
        self.cost_trace_overall = list()
        self.index_trace_overall = list()

        # reward related: initial/last
        self.a = a

        # (0816): newly added.
        self.measure = {"Workload Cost": list(),
                        "Reward": list(),
                        "Index Utility": list()}

    def checkout(self):
        """
        Select a good set of indexes in advance (Greedy).
        :return:
        """
        ratio = 0.4
        pre_is = list()
        while True:
            current_index = None
            current_max_x = 0
            # (0813): newly added.
            current_storage_assumption = 0
            start_sum = (np.array(self.pg_client2.get_queries_cost(self.workload)) * self.frequencies).sum()
            for no, index in enumerate(self.candidates):
                # (0813): newly added.
                if index in pre_is:
                    continue

                oid = self.pg_client2.execute_create_hypo(index)
                size = self.pg_client2.get_storage_cost([oid])[0]

                # (0813): newly added.
                if self.constraint == "storage" and current_storage_assumption + size > self.max_storage:
                    continue

                cu_sum = (np.array(self.pg_client2.get_queries_cost(self.workload)) * self.frequencies).sum()
                x = 1 - cu_sum / start_sum
                if x > ratio and x > current_max_x:
                    current_max_x = x
                    current_index = index
                    # (0813): newly added.
                    current_no = no
                    current_size = size

                self.pg_client2.execute_delete_hypo(oid)

            # (0813): newly added.
            if self.constraint == "number" and len(pre_is) >= self.max_count:
                break

            # exit when no more benefit
            if current_index is None:
                break

            pre_is.append(current_index)

            # (0813): newly added.
            self.index_no.append(current_no)
            self.sizes.append(current_size)
            current_storage_assumption += current_size

            self.pg_client2.execute_create_hypo(current_index)

        # pre_is = ['lineitem#l_orderkey,l_shipdate', 'lineitem#l_partkey,l_orderkey', 'lineitem#l_receiptdate', 'lineitem#l_shipdate,l_partkey', 'lineitem#l_suppkey,l_commitdate']
        # pre_is = ['lineitem#l_orderkey,l_suppkey', 'lineitem#l_partkey,l_suppkey', 'lineitem#l_receiptdate', 'lineitem#l_shipdate,l_discount', 'lineitem#l_suppkey,l_commitdate']
        # pre_is.append('lineitem#l_orderkey')
        self.pre_create = pre_is
        self.pg_client2.delete_indexes()

        return pre_is

    def reset(self):
        self.last_state = self.init_state
        self.last_cost = self.init_cost
        self.last_cost_sum = self.init_cost_sum

        # (0813): unused.
        self.performance_gain = np.zeros((len(self.candidates),), dtype=np.float)

        self.current_index_count = 0
        # (0818): newly added.
        self.current_storage_sum = 0
        self.current_index = np.zeros((len(self.candidates),), dtype=np.float)
        self.pg_client1.delete_indexes()

        if len(self.pre_create) > 0:
            for index in self.pre_create:
                self.pg_client1.execute_create_hypo(index)

            for no, size in zip(self.index_no, self.sizes):
                self.current_index[no] = 1

            self.current_index_count = len(self.pre_create)

            self.init_cost_sum = (np.array(self.pg_client1.get_queries_cost(self.workload)) * self.frequencies).sum()
            self.last_cost_sum = self.init_cost_sum

        return self.last_state

    def step(self, action):
        action = action[0]
        if self.current_index[action] != 0:
            # next_state, reward, done
            return self.last_state, 0, False

        # (0813): newly modified/added.
        oid = self.pg_client1.execute_create_hypo(self.candidates[action])
        storage_cost = self.pg_client1.get_storage_cost([oid])[0]
        if self.constraint == "storage" and \
                self.current_storage_sum + storage_cost > self.max_storage:
            self.pg_client1.execute_delete_hypo(oid)

            current_cost_info = np.array(self.pg_client1.get_queries_cost(self.workload)) * self.frequencies
            current_cost_sum = current_cost_info.sum()
            self.cost_trace_overall.append(current_cost_sum)
            self.index_trace_overall.append(self.current_index)

            self.measure["Index Utility"].append(1 - current_cost_sum / self.init_cost_sum)
            self.measure["Workload Cost"].append(current_cost_sum)

            return self.last_state, -1, True

        self.current_index[action] = 1

        self.current_storage_sum += storage_cost
        self.current_index_count += 1

        # reward & performance gain
        current_cost_info = np.array(self.pg_client1.get_queries_cost(self.workload)) * self.frequencies
        current_cost_sum = current_cost_info.sum()

        # update
        self.last_cost = current_cost_info
        # : to be modified.
        # state = (self.init_cost - current_cost_info) / self.init_cost
        self.last_state = np.append(self.frequencies, self.current_index)
        deltac0 = (self.init_cost_sum - current_cost_sum) / self.init_cost_sum
        deltac1 = (self.last_cost_sum - current_cost_sum) / self.init_cost_sum

        '''deltac0 = max(0.000003, deltac0)
        if deltac0 == 0.000003:
            reward = -10
        else:
            reward = math.log(0.0003, deltac0)'''

        b = 1 - self.a
        reward = self.a * deltac0 * 100 + b * deltac1 * 100
        # reward = deltac0
        # reward = math.log(0.99, deltac0)

        '''deltac0 = self.init_cost_sum/current_cost_sum
        deltac1 = self.last_cost_sum/current_cost_sum
        reward = math.log(deltac0,10)'''

        self.last_cost_sum = current_cost_sum
        # (0813): newly added.
        if self.constraint == "number" and self.current_index_count >= self.max_count:
            self.cost_trace_overall.append(current_cost_sum)
            self.index_trace_overall.append(self.current_index)

            self.measure["Index Utility"].append(1 - current_cost_sum / self.init_cost_sum)
            self.measure["Workload Cost"].append(current_cost_sum)

            return self.last_state, reward, True
        else:
            return self.last_state, reward, False
