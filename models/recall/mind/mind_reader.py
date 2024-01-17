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
import numpy as np
from paddle.io import IterableDataset
import random
import math


# random.seed(12345)


class RecDataset(IterableDataset):
    def __init__(self, file_list, config):
        super(RecDataset, self).__init__()
        self.file_list = file_list
        self.maxlen = config.get("hyper_parameters.maxlen", 30)
        self.batch_size = config.get("runner.train_batch_size", 128)
        self.batches_per_epoch = config.get("runner.batches_per_epoch", 1000)
        self.min_data_len = config.get("hyper_parameters.min_data_len", 1)
        self.sample_strategy = config.get("hyper_parameter.sample_strategy", 1)
        self.long_seq_strategy = config.get("hyper_parameter.long_seq_strategy", 1)
        self.ads_campaign_count = config.get("hyper_parameters.ads_campaign_count", 200)
        self.ads_group_count = config.get("hyper_parameters.ads_group_count", 300)
        self.brand_count = config.get("hyper_parameters.brand_count", 50)
        self.phone_model_count = config.get("hyper_parameters.phone_model_count", 1500)
        self.product_count = config.get("hyper_parameters.product_count", 15)
        self.item_count = config.get("hyper_parameters.item_count", 2000)
        self.shuffle_data = config.get("runner.shuffle_data", False)
        self.deduplicate_data = config.get("runner.deduplicate_data", False)
        self.sample_weights = config.get("runner.sample_weights", False)
        self.unk = 1

        self.init()
        self.count = 0

    def init(self):
        self.graph = {}
        self.item_graph = {}
        self.users = set()
        self.items = set()
        for file in self.file_list:
            with open(file, "r") as f:
                for line in f:
                    conts = line.strip().split(',')
                    user_id = int(conts[0])
                    item_id = int(conts[1])
                    user_country_id = int(conts[2])
                    item_country_id = int(conts[3])
                    ads_campaign_id = int(conts[4])
                    if ads_campaign_id >= self.ads_campaign_count:
                        ads_campaign_id = self.unk
                    ads_group_id = int(conts[5])
                    if ads_group_id >= self.ads_group_count:
                        ads_group_id = self.unk
                    brand_id = int(conts[6])
                    if brand_id >= self.brand_count:
                        brand_id = self.unk
                    height_id = int(conts[7])
                    phone_model_id = int(conts[8])
                    if phone_model_id >= self.phone_model_count:
                        phone_model_id = self.unk
                    product_id = int(conts[9])
                    if product_id >= self.product_count:
                        product_id = self.unk
                    if item_id >= self.item_count:
                        item_id = self.unk
                    time_stamp = int(conts[10])
                    self.users.add(user_id)
                    self.items.add(item_id)
                    if user_id not in self.graph:
                        self.graph[user_id] = []
                        self.item_graph[user_id] = []
                    if not self.deduplicate_data or (item_id not in self.item_graph[user_id]):
                        self.graph[user_id].append((item_id, time_stamp, user_country_id, item_country_id,
                                                    ads_campaign_id, ads_group_id, brand_id, height_id,
                                                    phone_model_id, product_id))
                        self.item_graph[user_id].append(item_id)
        for user_id, value in self.graph.items():
            value.sort(key=lambda x: x[1])
            self.graph[user_id] = [(x[0], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]) for x in value]
        self.users = list(self.users)
        self.items = list(self.items)
        self.weights = [math.floor(math.sqrt(len(self.graph[x]))) for x in self.users]

    def __iter__(self):
        # random.seed(12345)
        while True:
            if self.sample_weights:
                user_id_list = random.choices(self.users, weights=self.weights, k=self.batch_size)
            else:
                user_id_list = random.sample(self.users, self.batch_size)
            if self.count >= self.batches_per_epoch * self.batch_size:
                self.count = 0
                break
            for user_id in user_id_list:
                item_list = self.graph[user_id]
                if len(item_list) < self.min_data_len:
                    continue
                # random.seed(12345)

                if self.long_seq_strategy == 1:
                    times = 1
                else:
                    times = math.floor((len(item_list) + 1) / 4) + 1
                    times = min(times, 10)
                for i in range(0, times):
                    if self.sample_strategy == 1:
                        k = max(random.choice(range(0, len(item_list))), random.choice(range(0, len(item_list))))
                    else:
                        k = random.choice(range(0, len(item_list)))
                    if self.shuffle_data:
                        random.shuffle(item_list)
                    (item_id, user_country_id, item_country_id, ads_campaign_id, ads_group_id, brand_id, height_id,
                     phone_model_id, product_id) = item_list[k]

                    if k >= self.maxlen:
                        hist_item_list = [x[0] for x in item_list[k - self.maxlen:k]]
                        hist_country_list = [x[2] for x in item_list[k - self.maxlen:k]]
                        hist_item_len = len(hist_item_list)
                    else:
                        hist_item_list = [x[0] for x in item_list[0:k]] + [0] * (self.maxlen - k)
                        hist_country_list = [x[2] for x in item_list[0:k]] + [0] * (self.maxlen - k)
                        hist_item_len = k
                    self.count += 1
                    yield [
                        np.array(hist_item_list).astype("int64"),
                        np.array([item_id]).astype("int64"),
                        np.array([hist_item_len]).astype("int64"),
                        np.array(hist_country_list).astype("int64"),
                        np.array([user_country_id]).astype("int64"),
                        np.array([item_country_id]).astype("int64"),
                        np.array([ads_campaign_id]).astype("int64"),
                        np.array([ads_group_id]).astype("int64"),
                        np.array([brand_id]).astype("int64"),
                        np.array([height_id]).astype("int64"),
                        np.array([phone_model_id]).astype("int64"),
                        np.array([product_id]).astype("int64")
                    ]
