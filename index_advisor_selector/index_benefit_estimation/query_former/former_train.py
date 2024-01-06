import logging
import os
import json
import argparse

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from index_advisor_selector.index_benefit_estimation.query_former.model.model import QueryFormer

from index_advisor_selector.index_benefit_estimation.query_former.model import util
from index_advisor_selector.index_benefit_estimation.query_former.model.util import Normalizer, set_logger
from index_advisor_selector.index_benefit_estimation.query_former.model.util import seed_everything

from index_advisor_selector.index_benefit_estimation.query_former.model.dataset import PlanTreeDataset
from index_advisor_selector.index_benefit_estimation.query_former.model.trainer import eval_workload, train
from index_advisor_selector.index_benefit_estimation.query_former.model.database_util import get_hist_file, get_job_table_sample
from index_advisor_selector.index_benefit_estimation.benefit_utils.benefit_const import alias2table_tpch, alias2table_job, alias2table_tpcds


def get_parser():
    parser = argparse.ArgumentParser(
        description="A DEEP MODEL for Learned Plan Representation (QueryFormer).")

    parser.add_argument("--seed", type=int, default=666)

    # file params.
    parser.add_argument("--exp_id", type=str, default="exp_pre")
    parser.add_argument("--gpu_no", type=int, default=1)

    # parser.add_argument("--data_file", type=str,
    #                     default="./data/TPC_DS_10_by_query.csv")

    # tpch
    parser.add_argument("--train_data_file", type=str,
                        default="/data/wz/index/index_eab/eab_benefit/query_former/data/tpch/tpch_cost_data_tgt_train.json")
    parser.add_argument("--valid_data_file", type=str,
                        default="/data/wz/index/index_eab/eab_benefit/query_former/data/tpch/tpch_cost_data_tgt_valid.json")
    parser.add_argument("--test_data_file", type=str,
                        default="/data/wz/index/index_eab/eab_benefit/query_former/data/tpch/tpch_cost_data_tgt_test.json")

    parser.add_argument("--encoding_load", type=str,
                        default="/data/wz/index/index_eab/eab_benefit/query_former/data/tpch/encoding_tpch_v2.pt")
    parser.add_argument("--model_load", type=str,
                        default="/data/wz/index/index_eab/eab_benefit/query_former/exp_res/exp_former_tpch_tgt_ep500_bat1024/model/former_FORMER_200.pt")
    parser.add_argument("--cost_norm_load", type=str,
                        default="/data/wz/index/index_eab/eab_benefit/query_former/data/tpch/cost_norm_tpch_v2.pt")
    parser.add_argument("--card_norm_load", type=str,
                        default="/data/wz/index/index_eab/eab_benefit/query_former/data/tpch/cost_norm_tpch_v2.pt")

    # tpcds
    # parser.add_argument("--train_data_file", type=str,
    #                     default="/data/wz/index/index_eab/eab_benefit/query_former/data/tpcds/tpcds_cost_data_tgt_train.json")
    # parser.add_argument("--valid_data_file", type=str,
    #                     default="/data/wz/index/index_eab/eab_benefit/query_former/data/tpcds/tpcds_cost_data_tgt_valid.json")
    # parser.add_argument("--test_data_file", type=str,
    #                     default="/data/wz/index/index_eab/eab_benefit/query_former/data/tpcds/tpcds_cost_data_tgt_test.json")
    #
    # parser.add_argument("--model_load", type=str,
    #                     default="/data/wz/index/index_eab/eab_benefit/query_former/exp_res/exp_former_tpcds_tgt_ep500_bat1024/model/former_FORMER_200.pt")
    # parser.add_argument("--encoding_load", type=str,
    #                     default="/data/wz/index/index_eab/eab_benefit/query_former/data/tpcds/encoding_tpcds_v2.pt")
    # parser.add_argument("--cost_norm_load", type=str,
    #                     default="/data/wz/index/index_eab/eab_benefit/query_former/data/tpcds/cost_norm_tpcds_v2.pt")
    # parser.add_argument("--card_norm_load", type=str,
    #                     default="/data/wz/index/index_eab/eab_benefit/query_former/data/tpcds/cost_norm_tpcds_v2.pt")

    # job
    # parser.add_argument("--train_data_file", type=str,
    #                     default="/data/wz/index/index_eab/eab_benefit/query_former/data/job/job_cost_data_tgt_train.json")
    # parser.add_argument("--valid_data_file", type=str,
    #                     default="/data/wz/index/index_eab/eab_benefit/query_former/data/job/job_cost_data_tgt_valid.json")
    # parser.add_argument("--test_data_file", type=str,
    #                     default="/data/wz/index/index_eab/eab_benefit/query_former/data/job/job_cost_data_tgt_test.json")
    #
    # parser.add_argument("--model_load", type=str,
    #                     default="/data/wz/index/index_eab/eab_benefit/query_former/exp_res/exp_former_job_tgt_ep500_bat1024/model/former_FORMER_200.pt")
    # parser.add_argument("--encoding_load", type=str,
    #                     default="/data/wz/index/index_eab/eab_benefit/query_former/data/job/encoding_job_v2.pt")
    # parser.add_argument("--cost_norm_load", type=str,
    #                     default="/data/wz/index/index_eab/eab_benefit/query_former/data/job/cost_norm_job_v2.pt")
    # parser.add_argument("--card_norm_load", type=str,
    #                     default="/data/wz/index/index_eab/eab_benefit/query_former/data/job/cost_norm_job_v2.pt")

    parser.add_argument("--model_save", type=str,
                        default="./exp_res/{}/model/former_{}.pt")
    parser.add_argument("--model_save_gap", type=int, default=10)

    parser.add_argument("--runlog", type=str,
                        default="./exp_res/{}/exp_runtime.log")
    parser.add_argument("--logdir", type=str,
                        default="./exp_res/{}/logdir/")

    # running params.
    parser.add_argument("--epoch_num", type=int, default=200, help="number of the training epoch")
    parser.add_argument("--batch_size", type=int, default=1024, help="value of the mini batch-size")
    parser.add_argument("--lr", type=float, default=0.001, help="value of the learning rate")

    parser.add_argument("--use_sample", action="store_true")
    parser.add_argument("--use_hist", action="store_true")

    # model params.
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--clip_size", type=int, default=50)
    parser.add_argument("--embed_size", type=int, default=64)
    parser.add_argument("--pred_hid", type=int, default=128)
    parser.add_argument("--ffn_dim", type=int, default=128)
    parser.add_argument("--head_size", type=int, default=12)
    parser.add_argument("--sch_decay", type=float, default=0.6)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--to_predict", type=str, default="cost")

    return parser


def main(args):
    device = torch.device(f"cuda:{args.gpu_no}" if torch.cuda.is_available() else "cpu")

    # (1005): newly modified.
    # hist_file = pd.DataFrame()
    hist_file = None
    # data_path = "./data/imdb/"
    # hist_file = get_hist_file(data_path + "histogram_string.csv")

    # zw: np.log()? MinMaxScaler()?
    # (1005): newly modified.
    if os.path.exists(args.cost_norm_load):
        cost_norm = torch.load(args.cost_norm_load)
    else:
        cost_norm = Normalizer()

    if os.path.exists(args.card_norm_load):
        card_norm = torch.load(args.card_norm_load)
    else:
        card_norm = Normalizer()

    # cost_norm = Normalizer(-3.61192, 12.290855)
    # card_norm = Normalizer(1, 100)

    # col2idx, column_min_max_vals, idx2col, idx2join, idx2op,
    # idx2table, idx2type, join2idx, op2idx, table2idx, type2idx
    # (1005): newly modified.
    encoding = torch.load(args.encoding_load)
    # encoding_ckpt = torch.load("checkpoints/encoding.pt")
    # encoding = encoding_ckpt["encoding"]

    # imdb_path = "./data/imdb/"
    # full_train_df = pd.DataFrame()
    # # zw: num(18 * 5000 = 90000)
    # for i in range(18):  # 18
    #     file = imdb_path + "plan_and_cost/train_plan_part{}.csv".format(i)
    #     # zw: ['id', 'json']
    #     df = pd.read_csv(file)
    #     full_train_df = full_train_df.append(df)
    #
    # val_df = pd.DataFrame()
    # for i in range(18, 20):  # 18, 20
    #     file = imdb_path + "plan_and_cost/train_plan_part{}.csv".format(i)
    #     df = pd.read_csv(file)
    #     val_df = val_df.append(df)

    with open(args.train_data_file, "r") as rf:
        full_train_df = json.load(rf)
    with open(args.valid_data_file, "r") as rf:
        val_df = json.load(rf)

    # (1005): newly modified.
    table_sample = None
    # table_sample = get_job_table_sample(imdb_path + "train")

    # (1005): newly added.
    if "tpch" in args.train_data_file:
        alias2tbl = alias2table_tpch
    elif "tpcds" in args.train_data_file:
        alias2tbl = alias2table_tpcds
    elif "job" in args.train_data_file:
        alias2tbl = alias2table_job

    train_ds = PlanTreeDataset(full_train_df, None, encoding, hist_file,
                               card_norm, cost_norm, args.to_predict, table_sample, alias2tbl=alias2tbl)
    val_ds = PlanTreeDataset(val_df, None, encoding, hist_file,
                             card_norm, cost_norm, args.to_predict, table_sample, alias2tbl=alias2tbl)

    # (1007): newly added.
    torch.save(cost_norm, args.cost_norm_load.replace(".pt", "_v2.pt"))
    torch.save(card_norm, args.card_norm_load.replace(".pt", "_v2.pt"))

    logging.info(f"Load the train data from `{args.train_data_file}` ({len(train_ds)}).")
    logging.info(f"Load the valid data from `{args.valid_data_file}` ({len(val_ds)}).")

    encoding_save = args.encoding_load.replace(".pt", "_v2.pt")
    torch.save(encoding, encoding_save)

    model = QueryFormer(emb_size=args.embed_size, ffn_dim=args.ffn_dim, head_size=args.head_size,
                        dropout=args.dropout, n_layers=args.n_layers, encoding=encoding,
                        use_sample=args.use_sample, use_hist=args.use_hist, pred_hid=args.pred_hid)
    # if os.path.exists(args.model_load):
    #     # "checkpoints/cost_model.pt"
    #     checkpoint = torch.load(args.model_load, map_location="cpu")
    #     model.load_state_dict(checkpoint["model"])

    model.to(device)

    crit = nn.MSELoss()
    model, best_path = train(model, train_ds, val_ds, crit, cost_norm, args, device=device)

    print(f"The path of the best model is `{best_path}`.")

    # methods = {
    #     'get_sample': get_job_table_sample,
    #     'encoding': encoding,
    #     'cost_norm': cost_norm,
    #     'hist_file': hist_file,
    #     'model': model,
    #     'device': args.device,
    #     'bs': 512,
    # }

    # _ = eval_workload('job-light', methods)
    # _ = eval_workload('synthetic', methods)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # Setting random seed to facilitate reproduction
    seed_everything(args.seed)

    # logging / tensorboard
    if not os.path.exists(os.path.dirname(args.runlog.format(args.exp_id))):
        os.makedirs(os.path.dirname(args.runlog.format(args.exp_id)))
        os.makedirs(os.path.dirname(args.model_save.format(args.exp_id, 0)))
    set_logger(args.runlog.format(args.exp_id))

    util.tf_step = 0
    util.summary_writer = SummaryWriter(args.logdir.format(args.exp_id))
    util.summary_writer.add_text(
        "parameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|"
                                                 for key, value in vars(args).items()])),
        0
    )

    main(args)
