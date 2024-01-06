# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: lib_train
# @Author: Wei Zhou
# @Time: 2023/6/7 20:50

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from csv import reader

import json
import numpy as np
from tqdm import tqdm

import ast
import argparse
import logging

import torch
from torch.nn import MSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter

from index_advisor_selector.index_benefit_estimation.index_cost_lib.utils import lib_com
from index_advisor_selector.index_benefit_estimation.index_cost_lib.lib_data import collate_fn4lib
from index_advisor_selector.index_benefit_estimation.index_cost_lib.lib_model import make_model, self_attn_model, q_error


def get_parser():
    parser = argparse.ArgumentParser(
        description="A DEEP MODEL for Learned Index Benefits (LIB).")

    parser.add_argument("--seed", type=int, default=666)

    # file params.
    parser.add_argument("--exp_id", type=str, default="exp_ep500")
    parser.add_argument("--gpu_no", type=int, default=0)

    parser.add_argument("--data_file", type=str,
                        default="./data/TPC_DS_10_by_query.csv")

    # 1. tpch
    parser.add_argument("--train_data_file", type=str,
                        default="/data/wz/index/index_eab/eab_benefit/index_cost_lib/data/tpch/lib_tpch_cost_data_tgt_train.json")
    parser.add_argument("--valid_data_file", type=str,
                        default="/data/wz/index/index_eab/eab_benefit/index_cost_lib/data/tpch/lib_tpch_cost_data_tgt_valid.json")
    parser.add_argument("--test_data_file", type=str,
                        default="/data/wz/index/index_eab/eab_benefit/index_cost_lib/data/tpch/lib_tpch_cost_data_tgt_test.json")

    parser.add_argument("--model_load", type=str, default="/data/wz/index/index_eab/eab_benefit/index_cost_lib/exp_res/exp_lib_tpch_tgt_ep500_bat2048/model/lib_LIB_200.pt",
                        help="the path to the saved model")

    # 2. tpcds
    # parser.add_argument("--train_data_file", type=str,
    #                     default="/data/wz/index/index_eab/eab_benefit/index_cost_lib/data/tpcds/lib_tpcds_cost_data_tgt_train.json")
    # parser.add_argument("--valid_data_file", type=str,
    #                     default="/data/wz/index/index_eab/eab_benefit/index_cost_lib/data/tpcds/lib_tpcds_cost_data_tgt_valid.json")
    # parser.add_argument("--test_data_file", type=str,
    #                     default="/data/wz/index/index_eab/eab_benefit/index_cost_lib/data/tpcds/lib_tpcds_cost_data_tgt_test.json")
    #
    # parser.add_argument("--model_load", type=str,
    #                     default="/data/wz/index/index_eab/eab_benefit/index_cost_lib/exp_res/exp_lib_tpcds_tgt_ep500_bat2048/model/lib_LIB_200.pt",
    #                     help="the path to the saved model")

    # 3. job
    # parser.add_argument("--train_data_file", type=str,
    #                     default="/data/wz/index/index_eab/eab_benefit/index_cost_lib/data/job/lib_job_cost_data_tgt_train.json")
    # parser.add_argument("--valid_data_file", type=str,
    #                     default="/data/wz/index/index_eab/eab_benefit/index_cost_lib/data/job/lib_job_cost_data_tgt_valid.json")
    # parser.add_argument("--test_data_file", type=str,
    #                     default="/data/wz/index/index_eab/eab_benefit/index_cost_lib/data/job/lib_job_cost_data_tgt_test.json")
    #
    # parser.add_argument("--model_load", type=str,
    #                     default="/data/wz/index/index_eab/eab_benefit/index_cost_lib/exp_res/exp_lib_job_tgt_ep500_bat1024/model/lib_LIB_200.pt",
    #                     help="the path to the saved model")

    parser.add_argument("--runlog", type=str,
                        default="./exp_res/{}/exp_runtime.log")
    parser.add_argument("--logdir", type=str,
                        default="./exp_res/{}/logdir/")
    # parser.add_argument("--model_load", type=str, default="./model/LIB_query.pth",
    #                     help="the path to the saved model")
    parser.add_argument("--model_save", type=str,
                        default="./exp_res/{}/model/lib_{}.pt")
    parser.add_argument("--model_save_gap", type=int, default=10)

    # running params.
    parser.add_argument("--epoch_num", type=int, default=500, help="number of the training epoch")
    parser.add_argument("--batch_size", type=int, default=1024, help="value of the mini batch-size")
    parser.add_argument("--lr", type=float, default=0.001, help="value of the learning rate")

    # model params.
    parser.add_argument("--input_dim", type=int, default=12, help="the dimension of the input feature")
    parser.add_argument("--dim1", type=int, default=32, help="embedding size")
    parser.add_argument("--dim2", type=int, default=64, help="hidden dimension for prediction layer")
    parser.add_argument("--dim3", type=int, default=128, help="hidden dimension for FNN")
    parser.add_argument("--n_encoder_layers", type=int, default=6,
                        help="number of layer of attention encoder")
    parser.add_argument("--n_heads", type=int, default=8, help="number of heads in attention")
    parser.add_argument("--dropout_r", type=float, default=0.2, help="dropout ratio")

    return parser


def train(args):
    # cuda environment is recommended
    device = torch.device(f"cuda:{args.gpu_no}" if torch.cuda.is_available() else "cpu")

    # with open(args.data_file, "r") as ro:
    #     csv_reader = reader(ro)
    #     raw_data = list(csv_reader)
    #
    # # 12 = 5 + 1 + 1 + 1 + 1 + 2 + 1
    # # [𝑂𝑡, log(𝑐𝑎𝑟𝑑), log(𝑟𝑜𝑤𝑠), 𝑑𝑖𝑠𝑡_𝑓𝑟𝑎𝑐, 𝑁𝑈𝐿𝐿_𝑓𝑟𝑎𝑐, 𝐼𝑡, 𝐼𝑜]
    # data_label = list()
    # for i in range(0, len(raw_data[0])):
    #     lists = ast.literal_eval(raw_data[0][i])
    #     data_label.append(lists)
    #
    # train_data, valid_data = random_split(data_label, [int(0.8 * len(data_label)),
    #                                                    len(data_label) - int(0.8 * len(data_label))])

    with open(args.train_data_file, "r") as rf:
        train_data = json.load(rf)
    with open(args.valid_data_file, "r") as rf:
        valid_data = json.load(rf)

    # min(1, item["w/ actual cost"] / item["w/o actual cost"])
    train_data = [[item["feat"], item["w/ actual cost"] / item["w/o actual cost"]] for item in train_data
                  if len(item["feat"]) > 0 and item["w/ actual cost"] / item["w/o actual cost"] <= 1]
    valid_data = [[item["feat"], item["w/ actual cost"] / item["w/o actual cost"]] for item in valid_data
                  if len(item["feat"]) > 0 and item["w/ actual cost"] / item["w/o actual cost"] <= 1]

    logging.info(f"Load the train data from `{args.train_data_file}` ({len(train_data)}).")
    logging.info(f"Load the valid data from `{args.valid_data_file}` ({len(valid_data)}).")

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn4lib, drop_last=False)
    valid_loader = DataLoader(dataset=valid_data, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn4lib, drop_last=False)

    encoder_model, pooling_model = make_model(args.dim1, args.n_encoder_layers,
                                              args.dim3, args.n_heads, dropout=args.dropout_r)
    model = self_attn_model(encoder_model, pooling_model, args.input_dim, args.dim1, args.dim2)

    # model.load_state_dict(torch.load(args.model_path))
    if os.path.exists(args.model_load):
        checkpoint = torch.load(args.model_load, map_location="cpu")
        model.load_state_dict(checkpoint["model"])

    model = model.to(device)
    criterion = MSELoss()
    criterion = q_error

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5,
                                  patience=10, min_lr=1e-5, verbose=True)

    # model = torch.nn.DataParallel(model)
    # model.cuda()

    # model = model.to(device)
    # criterion = criterion.to(device)
    # criterion = cuda()

    for epoch in tqdm(range(1, args.epoch_num + 1)):
        # if (epoch) % config.lr_decay_epochs == 0:
        #     for g in optimizer.param_groups:
        #         g["lr"] = g["lr"] * config.lr_decay_ratio

        logging.info(f"The `lr` of EP{epoch} is `{optimizer.param_groups[0]['lr']}`.")

        model.train()
        total_loss = 0
        pro_bar = tqdm(enumerate(train_loader))
        for bi, batch in pro_bar:
            pro_bar.set_description(f"Epoch [{epoch}/{args.epoch_num}]")
            optimizer.zero_grad()

            pad_data, mask, label = batch
            pad_data, mask, label = pad_data.to(device), mask.to(device), label.to(device)
            pred_rr = model(pad_data, mask)

            loss = criterion(label, pred_rr)

            # optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # : newly added, scheduler.
            # scheduler.step(loss)

            total_loss += loss.item()
            pro_bar.set_postfix(train_loss=total_loss / (bi + 1))

            lib_com.add_summary_value("train loss", loss.item())
            lib_com.tf_step += 1
            if lib_com.tf_step % 100 == 0:
                lib_com.summary_writer.flush()
        logging.info(f"The final train loss of EP{epoch} is: {total_loss / (bi + 1)}.")

        model.eval()
        total_loss = 0
        pro_bar = tqdm(enumerate(valid_loader))
        for bi, batch in pro_bar:
            pro_bar.set_description(f"Epoch [{epoch}/{args.epoch_num}]")

            pad_data, mask, label = batch
            pad_data, mask, label = pad_data.to(device), mask.to(device), label.to(device)
            pred_rr = model(pad_data, mask)

            loss = criterion(label, pred_rr)

            total_loss += loss.item()
            pro_bar.set_postfix(valid_loss=total_loss / (bi + 1))

            lib_com.add_summary_value("valid loss", loss.item())
            lib_com.tf_step += 1
            if lib_com.tf_step % 100 == 0:
                lib_com.summary_writer.flush()

        # : newly added, scheduler.
        scheduler.step(total_loss / (bi + 1))

        logging.info(f"The final valid loss of EP{epoch} is: {total_loss / (bi + 1)}.")

        model_state_dict = model.state_dict()
        model_source = {
            "settings": args,
            "model": model_state_dict,
        }
        if epoch % args.model_save_gap == 0:
            torch.save(model_source, args.model_save.format(
                args.exp_id, "LIB_" + str(epoch)))


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # Setting random seed to facilitate reproduction
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # logging / tensorboard
    if not os.path.exists(os.path.dirname(args.runlog.format(args.exp_id))):
        os.makedirs(os.path.dirname(args.runlog.format(args.exp_id)))
        os.makedirs(os.path.dirname(args.model_save.format(args.exp_id, 0)))
    lib_com.set_logger(args.runlog.format(args.exp_id))

    lib_com.tf_step = 0
    lib_com.summary_writer = SummaryWriter(args.logdir.format(args.exp_id))
    lib_com.summary_writer.add_text(
        "parameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|"
                                                 for key, value in vars(args).items()])),
        0
    )

    train(args)
