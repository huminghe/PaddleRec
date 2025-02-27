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

import os
import logging

import numpy
import paddle
import sys
import numpy as np

__dir__ = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(__dir__)
sys.path.append(
    os.path.abspath(os.path.join(__dir__, '..', '..', '..', 'tools')))

from utils.save_load import save_model, load_model
from utils.utils_single import load_yaml, get_abs_model, create_data_loader, reset_auc, load_dy_model_class

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

config_yaml = "config_predict_paid.yaml"
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
maxlen = config.get("hyper_parameters.maxlen", 30)

author_country_map_path = config.get("runner.author_country_map_path", None)
author_id_map_path = config.get("runner.author_id_map_path", None)
country_id_map_path = config.get("runner.country_id_map_path", None)
brand_id_map_path = config.get("runner.brand_id_map_path", None)
ads_campaign_id_map_path = config.get("runner.ads_campaign_id_map_path", None)
ads_group_id_map_path = config.get("runner.ads_group_id_map_path", None)
phone_model_id_map_path = config.get("runner.phone_model_id_map_path", None)
phone_height_id_map_path = config.get("runner.phone_height_id_map_path", None)
product_id_map_path = config.get("runner.product_id_map_path", None)

ads_campaign_count = config.get("hyper_parameters.ads_campaign_count", None)
ads_group_count = config.get("hyper_parameters.ads_group_count", None)
brand_count = config.get("hyper_parameters.brand_count", None)
phone_model_count = config.get("hyper_parameters.phone_model_count", None)
author_count = config.get("hyper_parameters.item_count", None)
author_country_count = config.get("hyper_parameters.country_count", None)
product_count = config.get("hyper_parameters.product_count", None)

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
b = np.ascontiguousarray(numpy.transpose(dy_model.output_softmax_linear.weight.numpy()))

import faiss


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


UNK_ID = 1
PADDING_ID = 0


class Predictor:

    def __init__(self):
        self.faiss_index = faiss.IndexFlatIP(b.shape[-1])
        self.faiss_index.add(b)
        self.author_country_map, _ = get_map(author_country_map_path, author_country_count)
        self.author_id_map, self.reverse_author_id_map = get_map(author_id_map_path, author_count)
        self.country_id_map, _ = get_map(country_id_map_path)
        self.brand_id_map, _ = get_map(brand_id_map_path, brand_count)
        self.ads_campaign_id_map, _ = get_map(ads_campaign_id_map_path, ads_campaign_count)
        self.ads_group_id_map, _ = get_map(ads_group_id_map_path, ads_group_count)
        self.phone_model_id_map, _ = get_map(phone_model_id_map_path, phone_model_count)
        self.phone_height_id_map, _ = get_map(phone_height_id_map_path, phone_model_count)
        self.product_id_map, _ = get_map(product_id_map_path, product_count)
        self.result_author_id_map = {}

    def update_online_cg(self, author_list):
        logger.info("author list length: " + str(len(author_list)))
        author_id_list = []
        for i in range(len(author_list)):
            author_id = self.author_id_map.get(author_list[i], UNK_ID)
            if author_id != UNK_ID:
                author_id_list.append(author_id)
        online_b = b[author_id_list]
        # logger.info("online b: " + str(online_b))
        logger.info("online b length: " + str(len(online_b)))
        self.faiss_index = faiss.IndexFlatIP(online_b.shape[-1])
        self.faiss_index.add(online_b)
        for i in range(len(author_id_list)):
            self.result_author_id_map[i] = author_id_list[i]
        logger.info("result author id map: " + str(self.result_author_id_map))
        return

    def predict(self, batch_data, top_n, threshold):
        user_embs, _ = dy_model_class.infer_forward(dy_model, None,
                                                    batch_data, config)

        user_embs = user_embs.numpy()
        user_embs = np.reshape(user_embs, [-1, user_embs.shape[-1]])
        D, I = self.faiss_index.search(user_embs, top_n)
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

    def create_predict_data(self, author_list, history_country_list, country, ads_campaign, ads_group,
                            brand, phone_model, product):
        author_id_list = [self.author_id_map.get(x, UNK_ID) for x in author_list]
        author_country_list = [self.author_country_map.get(x.lower(), UNK_ID) for x in history_country_list]
        country_id = self.country_id_map.get(country.lower(), UNK_ID)
        ads_campaign_id = self.ads_campaign_id_map.get(ads_campaign, UNK_ID)
        ads_group_id = self.ads_group_id_map.get(ads_group, UNK_ID)
        brand_id = self.brand_id_map.get(brand.lower(), UNK_ID)
        phone_model_id = self.phone_model_id_map.get(phone_model.lower(), UNK_ID)
        phone_height_id = self.phone_height_id_map.get(phone_model.lower(), 2)
        product_id = self.product_id_map.get(product.lower(), UNK_ID)

        seq_lens = []
        output_list = []
        output_country_list = []
        user_country_list = []
        ads_campaign_list = []
        ads_group_list = []
        brand_list = []
        phone_height_list = []
        phone_model_list = []
        product_list = []

        length = len(author_id_list)
        seq_lens.append(min(maxlen, length))
        hist_item_list = author_id_list[-maxlen:] + [PADDING_ID] * max(0, maxlen - length)
        hist_country_list = author_country_list[-maxlen:] + [PADDING_ID] * max(0, maxlen - length)

        output_list.append(np.array([hist_item_list]).astype("int64"))
        output_country_list.append(np.array([hist_country_list]).astype("int64"))
        user_country_list.append(country_id)
        ads_campaign_list.append(ads_campaign_id)
        ads_group_list.append(ads_group_id)
        brand_list.append(brand_id)
        phone_height_list.append(phone_height_id)
        phone_model_list.append(phone_model_id)
        product_list.append(product_id)

        return output_list + [np.array([seq_lens]).astype("int64")] + output_country_list + [
            np.array([user_country_list]).astype("int64")] + [np.array([ads_campaign_list]).astype("int64")] + [
            np.array([ads_group_list]).astype("int64")] + [np.array([brand_list]).astype("int64")] + [
            np.array([phone_height_list]).astype("int64")] + [np.array([phone_model_list]).astype("int64")] + [
            np.array([product_list]).astype("int64")]

    def predict_author_result(self, author_list, country_list, country, ads_campaign, ads_group, brand, phone_model,
                              product, top_n):
        threshold = 0
        batch_data = self.create_predict_data(author_list, country_list, country, ads_campaign, ads_group, brand,
                                              phone_model, product)
        logger.info("batch data: " + str(batch_data))
        predict_result = self.predict(batch_data, top_n, threshold)
        author_info_list = [
            (self.reverse_author_id_map.get(self.result_author_id_map.get(x[0], UNK_ID), "0"), x[1] - threshold) for x
            in predict_result]
        return author_info_list
