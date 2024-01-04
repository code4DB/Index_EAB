# -*- coding: utf-8 -*-
# @Project: index_eab
# @Module: lib_com
# @Author: Wei Zhou
# @Time: 2023/6/7 21:32

import logging

tf_step = 0
summary_writer = None


def set_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # log to file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)


def add_summary_value(key, value, step=None):
    if step is None:
        summary_writer.add_scalar(key, value, tf_step)
    else:
        summary_writer.add_scalar(key, value, step)
