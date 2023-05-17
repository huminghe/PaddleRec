# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
test_data_dir = config.get("runner.test_data_dir", None)
print_interval = config.get("runner.print_interval", None)
model_load_path = config.get("runner.infer_load_path", "model_output")
start_epoch = config.get("runner.infer_start_epoch", 0)
end_epoch = config.get("runner.infer_end_epoch", 10)
batch_size = config.get("runner.infer_batch_size", None)
maxlen = config.get("hyper_parameters.maxlen", 30)
author_country_map_path = config.get("runner.author_country_map_path", None)
author_id_map_path = config.get("runner.author_id_map_path", None)
country_id_map_path = config.get("runner.country_id_map_path", None)
os.environ["CPU_NUM"] = str(config.get("runner.thread_num", 2))

logger.info("**************common.configs**********")
logger.info(
    "use_gpu: {}, test_data_dir: {}, start_epoch: {}, end_epoch: {}, print_interval: {}, model_load_path: {}".format(
        use_gpu, test_data_dir, start_epoch, end_epoch, print_interval, model_load_path))
logger.info("**************common.configs**********")

place = paddle.set_device('gpu' if use_gpu else 'cpu')

dy_model = dy_model_class.create_model(config)

epoch_id = start_epoch
logger.info("load model epoch {}".format(epoch_id))
model_path = os.path.join(model_load_path, str(epoch_id))
load_model(model_path, dy_model)
b = np.ascontiguousarray(numpy.transpose(dy_model.output_softmax_linear.weight.numpy()))

import faiss

if use_gpu:
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0
    faiss_index = faiss.GpuIndexFlatIP(res, b.shape[-1], flat_config)
    faiss_index.add(b)
else:
    faiss_index = faiss.IndexFlatIP(b.shape[-1])
    faiss_index.add(b)


def get_map(path):
    map_result = {}
    reverse_map_result = {}
    with open(path, "r") as rf:
        for line in rf:
            key, value = line.split("\t")
            map_result[key] = int(value)
            reverse_map_result[int(value)] = key
    return map_result, reverse_map_result


author_country_map, _ = get_map(author_country_map_path)
author_id_map, reverse_author_id_map = get_map(author_id_map_path)
country_id_map, _ = get_map(country_id_map_path)
UNK_ID = 1
PADDING_ID = 0


def predict(batch_data, top_n, threshold, logger):
    user_embs, _ = dy_model_class.infer_forward(dy_model, None,
                                                batch_data, config)

    user_embs = user_embs.numpy()
    user_embs = np.reshape(user_embs, [-1, user_embs.shape[-1]])
    D, I = faiss_index.search(user_embs, top_n)
    item_list_set = set()
    item_cor_list = []
    item_list = list(zip(np.reshape(I, -1), np.reshape(D, -1)))
    item_list.sort(key=lambda x: x[1], reverse=True)
    for j in range(len(item_list)):
        if item_list[j][0] not in item_list_set and item_list[j][0] != 0 and item_list[j][1] > threshold:
            item_list_set.add(item_list[j][0])
            item_cor_list.append(item_list[j])
            if len(item_list_set) >= top_n:
                break
    return item_cor_list


def create_predict_data(author_list, country):
    author_id_list = [author_id_map.get(x, UNK_ID) for x in author_list]
    author_country_list = [author_country_map.get(x, UNK_ID) for x in author_list]
    country_id = country_id_map.get(country.lower(), UNK_ID)

    seq_lens = []
    output_list = []
    output_country_list = []
    user_country_list = []

    length = len(author_id_list)
    seq_lens.append(min(maxlen, length))
    hist_item_list = author_id_list[-maxlen:] + [PADDING_ID] * max(0, maxlen - length)
    hist_country_list = author_country_list[-maxlen:] + [PADDING_ID] * max(0, maxlen - length)

    output_list.append(np.array([hist_item_list]).astype("int64"))
    output_country_list.append(np.array([hist_country_list]).astype("int64"))
    user_country_list.append(country_id)

    return output_list + [np.array([seq_lens]).astype("int64")] + output_country_list + [
        np.array([user_country_list]).astype("int64")]


def predict_author_result(author_list, country, top_n, logger):
    threshold = -0.75
    batch_data = create_predict_data(author_list, country)
    predict_result = predict(batch_data, top_n, threshold, logger)
    author_info_list = [(reverse_author_id_map.get(x[0], "0"), x[1] - threshold) for x in predict_result]
    return author_info_list
