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
model_load_path = config.get("runner.infer_load_path", "model_output")
start_epoch = config.get("runner.infer_start_epoch", 0)

country_id_map_path = config.get("runner.country_id_map_path", None)
brand_id_map_path = config.get("runner.brand_id_map_path", None)
campaign_id_map_path = config.get("runner.campaign_id_map_path", None)
group_id_map_path = config.get("runner.group_id_map_path", None)
phone_model_id_map_path = config.get("runner.phone_model_id_map_path", None)
phone_height_map_path = config.get("runner.phone_height_map_path", None)
accurate_user_map_path = config.get("runner.accurate_user_map_path", None)
moloco_user_map_path = config.get("runner.moloco_user_map_path", None)

item_count = config.get("hyper_parameters.sparse_feature_number", None)

BRAND_UNK = 420
ACCURATE_UNK = 9232
CAMPAIGN_UNK = 228
GROUP_UNK = 315
MODEL_UNK = 1119
MOLOCO_UNK = 9235
COUNTRY_UNK = 1
PADDING_ID = 0

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
    return map_result


def get_height_map(path):
    map_result = {}
    with open(path, "r") as rf:
        for line in rf:
            key, value = line.split("\t")
            v = float(value)
            map_result[key] = v
    return map_result


country_map = get_map(country_id_map_path, item_count)
brand_map = get_map(brand_id_map_path, item_count)
phone_model_map = get_map(phone_model_id_map_path, item_count)
campaign_id_map = get_map(campaign_id_map_path, item_count)
group_id_map = get_map(group_id_map_path, item_count)
is_accurate_user_map = get_map(accurate_user_map_path, item_count)
is_moloco_user_map = get_map(moloco_user_map_path, item_count)

phone_model_height_map = get_height_map(phone_height_map_path)


def append_dense_feature(dense_list, feature):
    feature = float(feature)
    feature = np.log(feature + 1)
    dense_list.append(feature)


def create_predict_data(start_num, purchase_pop_num, pop_up_buy_num, chat_num, video_call_click_num,
                        country, brand, model, campaign_id, group_id, is_accurate_user, is_moloco_user):
    country_id = country_map.get(country.lower(), COUNTRY_UNK)
    brand_id = brand_map.get(brand, BRAND_UNK)
    phone_model_id = phone_model_map.get(model, MODEL_UNK)
    campaign_id = campaign_id_map.get(campaign_id, CAMPAIGN_UNK)
    group_id = group_id_map.get(group_id, GROUP_UNK)
    accurate_user_id = is_accurate_user_map.get(is_accurate_user, ACCURATE_UNK)
    moloco_user_id = is_moloco_user_map.get(is_moloco_user, MOLOCO_UNK)

    phone_height = phone_model_height_map.get(model, 1500.0)

    output_list = []
    output_list.append(np.array([country_id]).astype("int64"))
    output_list.append(np.array([campaign_id]).astype("int64"))
    output_list.append(np.array([group_id]).astype("int64"))
    output_list.append(np.array([brand_id]).astype("int64"))
    output_list.append(np.array([phone_model_id]).astype("int64"))
    output_list.append(np.array([accurate_user_id]).astype("int64"))
    output_list.append(np.array([moloco_user_id]).astype("int64"))

    dense_list = []
    append_dense_feature(dense_list, start_num)
    append_dense_feature(dense_list, purchase_pop_num)
    append_dense_feature(dense_list, pop_up_buy_num)
    append_dense_feature(dense_list, chat_num)
    append_dense_feature(dense_list, video_call_click_num)
    append_dense_feature(dense_list, phone_height)

    output_list.append(np.array(dense_list).astype("float32"))

    return output_list


def predict(start_num, purchase_pop_num, pop_up_buy_num, chat_num, video_call_click_num,
            country, brand, model, campaign_id, group_id, is_accurate_user, is_moloco_user):
    batch_data = create_predict_data(start_num, purchase_pop_num, pop_up_buy_num, chat_num, video_call_click_num,
                                     country, brand, model, campaign_id, group_id, is_accurate_user, is_moloco_user)

    logger.info("batch data: " + str(batch_data))
    result = dy_model_class.predict_forward(dy_model, batch_data, config).numpy()
    logger.info("result: " + str(result))

    return result
