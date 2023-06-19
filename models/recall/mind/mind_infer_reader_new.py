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


class RecDataset(IterableDataset):
    def __init__(self, file_list, config):
        super(RecDataset, self).__init__()
        self.file_list = file_list
        self.maxlen = config.get("hyper_parameters.maxlen", 30)
        self.ads_group_count = config.get("hyper_parameters.ads_group_count", 200)
        self.brand_count = config.get("hyper_parameters.brand_count", 50)
        self.phone_model_count = config.get("hyper_parameters.phone_model_count", 1500)
        self.item_count = config.get("hyper_parameters.item_count", 2000)
        self.init()

    def init(self):
        padding = 0
        self.padding = padding
        unk = 1
        self.unk = unk

    def refine_item_id(self, x):
        if x >= self.item_count:
            x = self.unk
        return x

    def __iter__(self):
        for file in self.file_list:
            with open(file, "r") as rf:
                for line in rf:
                    hist_items, hist_countries, user_country, target_item, ads_group, brand, height, phone_model, time = line.strip().split(
                        "\t")

                    hist_item_list = [self.refine_item_id(int(x)) for x in hist_items.split(",")]
                    hist_country_list = [int(x) for x in hist_countries.split(",")]

                    # if len(hist_item_list) < 2:
                    #     continue

                    output_list = []
                    output_country_list = []
                    user_country_list = []
                    seq_lens = []
                    eval_list = []
                    ads_group_list = []
                    brand_list = []
                    height_list = []
                    phone_model_list = []

                    ads_group = int(ads_group)
                    if ads_group >= self.ads_group_count:
                        ads_group = self.unk
                    brand = int(brand)
                    if brand >= self.brand_count:
                        brand = self.unk
                    phone_model = int(phone_model)
                    if phone_model >= self.phone_model_count:
                        phone_model = self.unk

                    length = len(hist_item_list)
                    seq_lens.append(min(self.maxlen, length))
                    hist_item_list = hist_item_list[-self.maxlen:] + [self.padding] * max(0, self.maxlen - length)
                    hist_country_list = hist_country_list[-self.maxlen:] + [self.padding] * max(0, self.maxlen - length)

                    output_list.append(np.array(hist_item_list).astype("int64"))
                    output_country_list.append(np.array(hist_country_list).astype("int64"))
                    user_country_list.append(int(user_country))
                    eval_list.append([int(target_item)] + [self.padding] * max(0, self.maxlen - 1))

                    ads_group_list.append(ads_group)
                    brand_list.append(brand)
                    height_list.append(int(height))
                    phone_model_list.append(phone_model)

                    yield output_list + [np.array(seq_lens).astype("int64")] + output_country_list + [
                        np.array(user_country_list).astype("int64")] + [np.array(ads_group_list).astype("int64")] + [
                              np.array(brand_list).astype("int64")] + [np.array(height_list).astype("int64")] + [
                              np.array(phone_model_list).astype("int64")] + [np.array(eval_list).astype("int64")]
