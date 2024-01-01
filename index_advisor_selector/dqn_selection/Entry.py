import os
import pickle
import logging

import Model as model

from dqn_utils import Encoding as en
from dqn_utils import ParserForIndex as pi
from dqn_utils.Common import get_parser, set_logger, gen_cands

# CIKM20: 14
# Q1, Q3, Q4, Q5, Q6, Q8, Q9, Q10, Q12, Q14, Q17, Q18, Q20, Q21

# 'Q_ITERATION': the target_network update frequency
# 'U_ITERATION': the number of experience replay per episode
# 'LEARNING_START': the size of memory (trajectories) arrives at certain number
conf = {"LR": 0.002, "EPSILON": 0.97, "Q_ITERATION": 200, "U_ITERATION": 5, "BATCH_SIZE": 64,
        "GAMMA": 0.95, "EPISODES": 10, "LEARNING_START": 600, "DECAY_EP": 50, "MEMORY_CAPACITY": 20000}


def One_Run_DQN(args, conf, is_dnn, is_ps, is_double, a):
    # if args.constraint == "number":
    #     conf["NAME"] = f"DQN_{args.constraint}_{args.max_count}"
    # elif args.constraint == "storage":
    #     conf["NAME"] = f"DQN_{args.constraint}_{args.max_storage}"
    # args.exp_id = conf["NAME"]

    conf["NAME"] = args.exp_id
    conf["EPISODES"] = args.epoch

    index_mode = "hypo"
    if not os.path.exists(os.path.dirname(args.logdir.format(args.exp_id))):
        os.makedirs(os.path.dirname(args.logdir.format(args.exp_id)))
    if not os.path.exists(os.path.dirname(args.model_save.format(args.exp_id, 0))):
        os.makedirs(os.path.dirname(args.model_save.format(args.exp_id, 0)))
    set_logger(args.runlog.format(args.exp_id))

    logging.info(f"Load workload from `{args.work_load}`.")

    if args.work_load.endswith(".pickle"):
        with open(args.work_load, "rb") as rf:
            workload = pickle.load(rf)
    elif args.work_load.endswith(".sql"):
        with open(args.work_load, "r") as rf:
            workload = rf.readlines()
    frequency = [1265, 897, 643, 1190, 521, 1688, 778, 1999, 1690, 1433, 1796, 1266, 1046, 1353]
    frequency = [1 for _ in range(len(workload))]

    if os.path.exists(args.cand_load):
        logging.info(f"Load candidate from `{args.cand_load}`.")
        with open(args.cand_load, "rb") as rf:
            index_candidates = pickle.load(rf)
    else:
        logging.info(f"Generate candidate based on `{args.work_load}`.")
        enc = en.encoding_schema(args.conf_load)
        sql_parser = pi.Parser(enc["attr"])

        index_candidates = gen_cands(workload, sql_parser)

    # params: a
    # deltac0 = (self.init_cost_sum - current_cost_sum) / self.init_cost_sum
    # deltac1 = (self.last_cost_sum - current_cost_sum) / self.init_cost_sum
    # b = 1 - self.a
    # reward = self.a * deltac0 * 100 + b * deltac1 * 100

    agent = model.DQN(args, workload, frequency, index_candidates, index_mode,
                      conf, is_dnn, is_ps, is_double, a)
    _indexes = agent.train()

    indexes = list()
    for _i, _idx in enumerate(_indexes):
        if _idx == 1.0:
            indexes.append(index_candidates[_i])

    return indexes


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # todo(0813): newly added.
    logging.disable(logging.DEBUG)

    # args, conf, is_dnn, is_ps, is_double, a
    indexes = One_Run_DQN(args, conf, False, True, True, 0)
    print(indexes)
