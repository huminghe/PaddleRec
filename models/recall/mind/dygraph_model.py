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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import net
import numpy as np


class DygraphModel():
    """DygraphModel
    """

    def create_model(self, config):
        item_count = config.get("hyper_parameters.item_count", None)
        country_count = config.get("hyper_parameters.country_count", None)
        user_country_count = config.get("hyper_parameters.user_country_count", None)
        ads_group_count = config.get("hyper_parameters.ads_group_count", None)
        brand_count = config.get("hyper_parameters.brand_count", None)
        height_count = config.get("hyper_parameters.height_count", None)
        phone_model_count = config.get("hyper_parameters.phone_model_count", None)
        embedding_dim = config.get("hyper_parameters.embedding_dim", 64)
        hidden_size = config.get("hyper_parameters.hidden_size", 64)
        neg_samples = config.get("hyper_parameters.neg_samples", 100)
        maxlen = config.get("hyper_parameters.maxlen", 30)
        pow_p = config.get("hyper_parameters.pow_p", 1.0)
        capsual_iters = config.get("hyper_parameters.capsual.iters", 3)
        capsual_max_k = config.get("hyper_parameters.capsual.max_k", 4)
        capsual_init_std = config.get("hyper_parameters.capsual.init_std", 1.0)
        dropout = config.get("hyper_parameters.dropout", 0.2)
        more_features = config.get("hyper_parameters.more_features", False)
        more_dropout = config.get("hyper_parameters.more_dropout", False)
        MIND_model = net.MindLayer(item_count, country_count, user_country_count, ads_group_count, brand_count,
                                   height_count, phone_model_count, embedding_dim, hidden_size, neg_samples, maxlen,
                                   pow_p, capsual_iters, capsual_max_k, capsual_init_std, dropout, more_features,
                                   more_dropout)
        return MIND_model

    # define feeds which convert numpy of batch data to paddle.tensor 
    def create_feeds_train(self, batch_data):
        # print(batch_data)
        hist_item = paddle.to_tensor(batch_data[0], dtype="int64")
        target_item = paddle.to_tensor(batch_data[1], dtype="int64")
        seq_len = paddle.to_tensor(batch_data[2], dtype="int64")
        hist_country = paddle.to_tensor(batch_data[3], dtype="int64")
        user_country = paddle.to_tensor(batch_data[4], dtype="int64")
        item_country = paddle.to_tensor(batch_data[5], dtype="int64")
        ads_group = paddle.to_tensor(batch_data[6], dtype="int64")
        brand = paddle.to_tensor(batch_data[7], dtype="int64")
        height_id = paddle.to_tensor(batch_data[8], dtype="int64")
        phone_model_id = paddle.to_tensor(batch_data[9], dtype="int64")
        return [hist_item, target_item, seq_len, hist_country, user_country, item_country, ads_group, brand, height_id,
                phone_model_id]

    # create_feeds_infer
    def create_feeds_infer(self, batch_data):
        batch_size = batch_data[0].shape[0]
        hist_item = paddle.to_tensor(batch_data[0], dtype="int64")
        target_item = paddle.zeros((batch_size, 1), dtype="int64")
        seq_len = paddle.to_tensor(batch_data[1], dtype="int64")
        hist_country = paddle.to_tensor(batch_data[2], dtype="int64")
        user_country = paddle.to_tensor(batch_data[3], dtype="int64")
        ads_group = paddle.to_tensor(batch_data[4], dtype="int64")
        brand = paddle.to_tensor(batch_data[5], dtype="int64")
        height_id = paddle.to_tensor(batch_data[6], dtype="int64")
        phone_model_id = paddle.to_tensor(batch_data[7], dtype="int64")
        return [hist_item, target_item, seq_len, hist_country, user_country, ads_group, brand, height_id,
                phone_model_id]

    # define optimizer 
    def create_loss(self, hit_prob):
        return paddle.mean(hit_prob)

    # define optimizer 
    def create_optimizer(self, dy_model, config):
        lr = config.get("hyper_parameters.optimizer.learning_rate", 0.001)
        optimizer = paddle.optimizer.Adam(
            learning_rate=lr, parameters=dy_model.parameters())
        return optimizer

    # define metrics such as auc/acc
    # multi-task need to define multi metric
    def create_metrics(self):
        metrics_list_name = []
        metrics_list = []
        return metrics_list, metrics_list_name

    # construct train forward phase  
    def train_forward(self, dy_model, metrics_list, batch_data, config):
        hist_item, labels, seqlen, hist_country, user_country, item_country, ads_group, brand, height_id, phone_model_id = self.create_feeds_train(
            batch_data)
        loss, weight, _, _, _ = dy_model.forward(hist_item, seqlen, labels, hist_country, user_country, item_country,
                                                 ads_group, brand, height_id, phone_model_id)
        loss = self.create_loss(loss)
        print_dict = {"loss": loss}
        return loss, metrics_list, print_dict

    # construct infer forward phase  
    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        hist_item, labels, seqlen, hist_country, user_country, ads_group, brand, height_id, phone_model_id = self.create_feeds_infer(
            batch_data)
        dy_model.eval()
        user_cap, cap_weight = dy_model.forward(hist_item, seqlen, labels, hist_country, user_country, None, ads_group,
                                                brand, height_id, phone_model_id)
        # update metrics
        print_dict = None
        return user_cap, cap_weight
