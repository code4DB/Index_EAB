# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: cost_data
# @Author: Wei Zhou
# @Time: 2022/8/31 10:59

import torch
import numpy as np


def filter_label(data_load, data_save):
    """
    data_file_id:
        "/data/wz/index/attack/data_resource/visrew_qplan/server103
        /103prenc_pgs200_plan2_filter_split_format_vec_woindex_res4755.pt"
    data_distribution(cost_ratio):
        mean         min            25th     50th    75th    max
        19.5929      4.7849e-07     0.0305   0.8952  1.0090  67066.4761
    data_distribution(cost_reduct):
        mean         min            25th     50th    75th    max
        -18.5929     -67065.4761    -0.0090  0.1047  0.9694  0.9999

    data_file_id:
        "/data/wz/index/attack/data_resource/visrew_qplan/server103
        /103prenc_pgs200_plan4_filter_split_vec_ran2w_res4755.pt"
    data_distribution(cost_ratio):
        mean         min            25th     50th    75th    max
        25941.0867   4.1520e-08     0.0743   0.9503  11.8938 24140126.9999
    data_distribution(cost_reduct):
        mean         min            25th     50th    75th    max
        -25940.0867  -24140125.9999 -10.8938 0.0496  0.9256  0.9999
    """

    data = torch.load(data_load)

    labels = [dat[1]["act_cost"] / dat[0]["act_cost"] for dat in data]
    # labels = [1 - dat[1]["act_cost"] / dat[0]["act_cost"] for dat in data]

    item = list()
    for dat in data:
        if dat[1]["act_cost"] / dat[0]["act_cost"] <= 20:
            item.append(dat)

    torch.save(item, data_save)


if __name__ == "__main__":
    data_load = "/data/wz/index/attack/data_resource/visrew_qplan/server103/" \
                "103prenc_pgs200_plan2_filter_split_format_vec_woindex_res4755.pt"
    data_load = "/data/wz/index/attack/data_resource/visrew_qplan/server103/" \
                "103prenc_pgs200_plan4_filter_split_vec_ran2w_res4755.pt"

    data_save = "/data/wz/index/attack/data_resource/visrew_qplan/server103/" \
                "103prenc_pgs200_plan4_filter20cr_split_vec_ran2w_res4755.pt"

    filter_label(data_load, data_save)
