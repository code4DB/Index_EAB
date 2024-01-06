# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: cost_data
# @Author: Wei Zhou
# @Time: 2022/8/31 10:59

import torch
import numpy as np


def filter_label(data_load, data_save):
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
