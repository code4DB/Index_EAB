# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: tree_cost_run
# @Author: Wei Zhou
# @Time: 2022/8/17 15:41

import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.linear_model import LinearRegression

from index_advisor_selector.index_benefit_estimation.tree_model.tree_cost_utils.tree_cost_loss import QError, RatioLoss, cal_mape
from index_advisor_selector.index_benefit_estimation.tree_model.tree_cost_utils import tree_cost_com


def optimizer_est():
    """
    29043.666582027512
    23306.98967826577
    51983.05700714619
    :return:
    """
    data_load = "/data/wz/index/attack/data_resource/visrew_qplan/server103/103prenc_pgs200_plan_filter_split_format_res4755.pt"
    data_load = "/data/wz/index/attack/data_resource/visrew_qplan/server103/103prenc_pgs200_plan2_filter_split_format_vec_woindex_res4755.pt"
    data_load = "/data/wz/index/attack/data_resource/visrew_qplan/server103/103prenc_pgs200_plan4_filter_split_vec_ran2w_res4755.pt"
    data_load = "/data/wz/index/attack/data_resource/visrew_qplan/server103/103prenc_pgs200_plan4_filter20cr_split_vec_ran2w_res4755.pt"
    data = torch.load(data_load)

    train_set, valid_set = random_split(data, [int(0.8 * len(data)), len(data) - int(0.8 * len(data))])
    train_set, valid_set = list(train_set), list(valid_set)
    if len(data[0]) > 1:
        est = np.array([dat[1]["est_cost"] / dat[0]["est_cost"] for dat in train_set])
        act = np.array([dat[1]["act_cost"] / dat[0]["act_cost"] for dat in train_set])
        mape = cal_mape(est, act)
        print(mape)

        est = np.array([dat[1]["est_cost"] / dat[0]["est_cost"] for dat in valid_set])
        act = np.array([dat[1]["act_cost"] / dat[0]["act_cost"] for dat in valid_set])
        mape = cal_mape(est, act)
        print(mape)
    elif len(data[0]) == 1:
        est = np.array([dat["est_cost"] for dat in train_set])
        act = np.array([dat["act_cost"] for dat in train_set])

        model = LinearRegression()
        model.fit(est, act)

        est = np.array([dat["est_cost"] for dat in valid_set])
        act = np.array([dat["act_cost"] for dat in valid_set])
        y_pred = model.predict(est)
        mape = cal_mape(y_pred, act)
        print(mape)


def cost_train(model, train_loader, valid_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # criterion = nn.BCELoss()  # reduction='sum'
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    # criterion = QError()
    # criterion = RatioLoss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    # : newly added, learning rate decay.
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5,
                                  patience=20, min_lr=1e-5, verbose=True)

    model = model.to(device)
    criterion = criterion.to(device)

    for epoch in tqdm(range(1, args.epoch + 1)):
        # learning rate: `lr` decay.
        # if (epoch) % config.lr_decay_epochs == 0:
        #     for g in optimizer.param_groups:
        #         g["lr"] = g["lr"] * config.lr_decay_ratio

        logging.info(f"The `lr` of EP{epoch} is `{optimizer.param_groups[0]['lr']}`.")

        model.train()
        total_loss = 0
        pro_bar = tqdm(enumerate(train_loader))
        pro_bar.set_description(f"Epoch [{epoch}/{args.epoch}]")
        for bi, batch in pro_bar:
            # pro_bar.set_description(f"Epoch [{epoch}/{args.epoch}]")
            feat, label = batch
            feat, label = feat.to(device), label.to(device)

            props = model(feat)
            loss = criterion(props, label.unsqueeze(-1))

            optimizer.zero_grad()
            loss.backward()
            # # : newly added, gradient clipping.
            # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10, norm_type=2)
            optimizer.step()

            # : newly added, scheduler.
            # scheduler.step(loss)

            total_loss += loss.item()
            pro_bar.set_postfix(train_loss=total_loss / (bi + 1))

            tree_cost_com.add_summary_value("train loss", loss.item())
            tree_cost_com.tf_step += 1
            if tree_cost_com.tf_step % 100 == 0:
                tree_cost_com.summary_writer.flush()
        logging.info(f"The final train loss of EP{epoch} is: {total_loss / (bi + 1)}.")

        model.eval()
        total_loss = 0
        pro_bar = tqdm(enumerate(valid_loader))
        for bi, batch in pro_bar:
            pro_bar.set_description(f"Epoch [{epoch}/{args.epoch}]")
            feat, label = batch
            feat, label = feat.to(device), label.to(device)

            # : whether apply `mask` (sql_tokens=None) to guarantee the validity.
            props = model(feat)
            loss = criterion(props, label.unsqueeze(-1))

            total_loss += loss.item()
            pro_bar.set_postfix(valid_loss=total_loss / (bi + 1))

            tree_cost_com.add_summary_value("train valid loss", loss.item())
            tree_cost_com.tf_step += 1
            if tree_cost_com.tf_step % 100 == 0:
                tree_cost_com.summary_writer.flush()

        # : newly added, scheduler.
        scheduler.step(total_loss / (bi + 1))
        logging.info(f"The final valid loss of EP{epoch} is: {total_loss / (bi + 1)}.")

        model_state_dict = model.state_dict()
        model_source = {
            "settings": args,
            # : together with args.
            "input_dim": model.inp_dim,
            "hidden_dim": model.hid_dim,
            "model": model_state_dict
        }
        if epoch % args.model_save_gap == 0:
            torch.save(model_source, args.model_save.format(
                args.exp_id, "train_" + str(epoch)))
