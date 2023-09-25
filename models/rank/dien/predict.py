# !/usr/bin/python
# -*- coding: utf-8 -*-
# @author: huminghe
# @date: 2023/9/24
#

from __future__ import print_function
import argparse
import time

import os
import warnings
import logging

import numpy
import paddle
import paddle.nn.functional as F
import sys
import numpy as np
from numpy.linalg import norm
import math

__dir__ = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(__dir__)
sys.path.append(
    os.path.abspath(os.path.join(__dir__, '..', '..', '..', 'tools')))

from utils.save_load import save_model, load_model
from utils.utils_single import load_yaml, get_abs_model, create_data_loader, reset_auc, load_dy_model_class

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

config_yaml = "config_predict.yaml"
abs_dir = os.path.dirname(os.path.abspath(config_yaml))
config_yaml = get_abs_model(config_yaml)

# load config
config = load_yaml(config_yaml)
config["config_abs_dir"] = abs_dir
# load static model class
dy_model_class = load_dy_model_class(config)

use_gpu = config.get("runner.use_gpu", True)
model_load_path = config.get("runner.infer_load_path", "model_output")
start_epoch = config.get("runner.infer_start_epoch", 0)
maxlen = config.get("hyper_parameters.maxlen", 50)

author_country_map_path = config.get("runner.author_country_map_path", None)
author_id_map_path = config.get("runner.author_id_map_path", None)

author_count = config.get("hyper_parameters.item_count", None)

os.environ["CPU_NUM"] = str(config.get("runner.thread_num", 2))

logger.info("**************common.configs**********")
logger.info(
    "use_gpu: {}, start_epoch: {}, model_load_path: {}".format(use_gpu, start_epoch, model_load_path))
logger.info("**************common.configs**********")

place = paddle.set_device('gpu' if use_gpu else 'cpu')

dy_model = dy_model_class.create_model(config)

epoch_id = start_epoch
logger.info("load model epoch {}".format(epoch_id))
model_path = os.path.join(model_load_path, str(epoch_id))
load_model(model_path, dy_model)
dy_model.eval()


def get_map(path, num=999999):
    map_result = {}
    reverse_map_result = {}
    with open(path, "r") as rf:
        for line in rf:
            key, value = line.split("\t")
            v = int(value)
            if v < num:
                map_result[key] = v
                reverse_map_result[v] = key
    return map_result, reverse_map_result


author_country_map, _ = get_map(author_country_map_path)
author_id_map, reverse_author_id_map = get_map(author_id_map_path, author_count)
UNK_ID = 1
PADDING_ID = 0


def predict(batch_data):
    predict = dy_model_class.predict_forward(dy_model, batch_data, config)

    return predict


def create_predict_data(author_list, candidate_list):
    author_id_list = [author_id_map.get(x, UNK_ID) for x in author_list]
    author_country_list = [author_country_map.get(x, UNK_ID) for x in author_list]

    candidate_id_list = [author_id_map.get(x, UNK_ID) for x in candidate_list]
    candidate_country_id_list = [author_country_map.get(x, UNK_ID) for x in candidate_list]

    max_len = len(author_id_list)
    if max_len <= 1:
        max_len = 1

    itemInput = [author_id_list for _ in candidate_id_list]
    itemRes0 = np.array(
        [x + [0] * (max_len - len(x)) for x in itemInput])
    item = itemRes0.astype("int64").reshape([-1, max_len])
    catInput = [author_country_list for x in candidate_id_list]
    catRes0 = np.array(
        [x + [0] * (max_len - len(x)) for x in catInput])
    cat = catRes0.astype("int64").reshape([-1, max_len])
    len_array = [max_len for _ in candidate_id_list]
    mask = np.array(
        [[0] * x + [-1e9] * (max_len - x)
         for x in len_array]).reshape([-1, max_len, 1])
    target_item_seq = np.array(
        [[x] * max_len for x in candidate_id_list]).astype("int64").reshape(
        [-1, max_len])
    target_cat_seq = np.array(
        [[x] * max_len for x in candidate_country_id_list]).astype("int64").reshape(
        [-1, max_len])

    res = []
    res.append(np.array(item))
    res.append(np.array(cat))
    res.append(np.array(candidate_id_list).astype('int64'))
    res.append(np.array(candidate_country_id_list).astype('int64'))
    res.append(np.array(mask).astype('float32'))
    res.append(np.array(target_item_seq))
    res.append(np.array(target_cat_seq))

    return res


def predict_author_result(author_list, candidate_list):
    batch_data = create_predict_data(author_list, candidate_list)
    # logger.info("batch data: " + str(batch_data))
    predict_result = predict(batch_data)
    logger.info("predict result: " + str(predict_result))

    author_score_result = []
    for i in range(len(candidate_list)):
        author_score_result.append((candidate_list[i], predict_result[i][0]))

    return author_score_result
