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
import numpy as np


class Mind_SampledSoftmaxLoss_Layer(nn.Layer):
    """SampledSoftmaxLoss with LogUniformSampler
    """

    def __init__(self,
                 num_classes,
                 n_sample,
                 unique=True,
                 remove_accidental_hits=True,
                 subtract_log_q=True,
                 num_true=1,
                 batch_size=None):
        super(Mind_SampledSoftmaxLoss_Layer, self).__init__()
        self.range_max = num_classes
        self.n_sample = n_sample
        self.unique = unique
        self.remove_accidental_hits = remove_accidental_hits
        self.subtract_log_q = subtract_log_q
        self.num_true = num_true
        self.prob = np.array([0.0] * self.range_max)
        self.batch_size = batch_size
        for i in range(1, self.range_max):
            self.prob[i] = (np.log(i + 2) - np.log(i + 1)) / np.log(self.range_max + 1)
        self.new_prob = paddle.assign(self.prob.astype("float32"))
        self.log_q = paddle.log(-(paddle.exp((-paddle.log1p(self.new_prob) * 2
                                              * n_sample)) - 1.0))
        self.loss = nn.CrossEntropyLoss(soft_label=True)

    def sample(self, labels):
        """Random sample neg_samples
        """
        n_sample = self.n_sample
        n_tries = 1 * n_sample
        neg_samples = paddle.multinomial(
            self.new_prob,
            num_samples=n_sample,
            replacement=self.unique is False)
        true_log_probs = paddle.gather(self.log_q, labels)
        samp_log_probs = paddle.gather(self.log_q, neg_samples)
        return true_log_probs, samp_log_probs, neg_samples

    def forward(self, inputs, labels, weights, bias):
        """forward
        """
        # weights.stop_gradient = False
        embedding_dim = paddle.shape(weights)[-1]
        true_log_probs, samp_log_probs, neg_samples = self.sample(labels)
        # print(neg_samples)
        n_sample = neg_samples.shape[0]

        b1 = paddle.shape(labels)[0]
        b2 = paddle.shape(labels)[1]

        all_ids = paddle.concat([labels.reshape((-1,)), neg_samples])
        all_w = paddle.gather(weights, all_ids)

        true_w = all_w[:-n_sample].reshape((-1, b2, embedding_dim))
        sample_w = all_w[-n_sample:].reshape((n_sample, embedding_dim))

        all_b = paddle.gather(bias, all_ids)
        true_b = all_b[:-n_sample].reshape((-1, 1))

        sample_b = all_b[-n_sample:]

        # [B, D] * [B, 1,D]
        true_logist = paddle.sum(paddle.multiply(true_w, inputs.unsqueeze(1)),
                                 axis=-1) + true_b
        # print(true_logist)

        sample_logist = paddle.matmul(
            inputs, sample_w, transpose_y=True) + sample_b

        if self.remove_accidental_hits:
            hit = (paddle.equal(labels[:, :], neg_samples))
            padding = paddle.ones_like(sample_logist) * -1e30
            sample_logist = paddle.where(hit, padding, sample_logist)

        if self.subtract_log_q:
            true_logist = true_logist - true_log_probs.unsqueeze(1)
            sample_logist = sample_logist - samp_log_probs

        out_logist = paddle.concat([true_logist, sample_logist], axis=1)
        out_label = paddle.concat(
            [
                paddle.ones_like(true_logist) / self.num_true,
                paddle.zeros_like(sample_logist)
            ],
            axis=1)
        out_label.stop_gradient = True

        loss = self.loss(out_logist, out_label)
        return loss, out_logist, out_label


class Mind_Capsual_Layer(nn.Layer):
    """Mind_Capsual_Layer
    """

    def __init__(self,
                 input_units,
                 output_units,
                 iters=3,
                 maxlen=32,
                 k_max=3,
                 init_std=1.0,
                 dropout=0.2,
                 more_dropout=False,
                 batch_size=None):
        super(Mind_Capsual_Layer, self).__init__()

        self.iters = iters
        self.input_units = input_units
        self.output_units = output_units
        self.maxlen = maxlen
        self.init_std = init_std
        self.k_max = k_max
        self.batch_size = batch_size
        self.dropout = nn.Dropout(p=dropout)
        self.more_dropout = more_dropout

        # B2I routing
        self.routing_logits = self.create_parameter(
            shape=[1, self.k_max, self.maxlen],
            attr=paddle.ParamAttr(
                name="routing_logits", trainable=False),
            default_initializer=nn.initializer.Normal(
                mean=0.0, std=self.init_std))

        # bilinear mapping
        self.bilinear_mapping_matrix = self.create_parameter(
            shape=[self.input_units, self.output_units],
            attr=paddle.ParamAttr(
                name="bilinear_mapping_matrix", trainable=True),
            default_initializer=nn.initializer.Normal(
                mean=0.0, std=self.init_std))
        self.relu_layer = nn.Linear(self.output_units * 2, self.output_units)

    def squash(self, Z):
        """squash
        """
        vec_squared_norm = paddle.sum(paddle.square(Z), axis=-1, keepdim=True)
        scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / paddle.sqrt(vec_squared_norm + 1e-8)
        vec_squashed = scalar_factor * Z
        return vec_squashed

    def sequence_mask(self, lengths, maxlen=None, dtype="bool"):
        """sequence_mask
        """
        batch_size = paddle.shape(lengths)[0]
        if maxlen is None:
            maxlen = lengths.max()
        row_vector = paddle.arange(
            0, maxlen,
            1).unsqueeze(0).expand(shape=(batch_size, maxlen)).reshape((batch_size, -1, maxlen))
        lengths = lengths.unsqueeze(-1)
        mask = row_vector < lengths
        return mask.astype(dtype)

    def forward(self, item_his_emb, seq_len, user_features_emb):
        """forward

        Args:
            item_his_emb : [B, seqlen, dim]
            seq_len : [B, 1]
            user_features_emb: [B, dim]
        """
        batch_size = item_his_emb.shape[0]
        seq_len_tile = paddle.tile(seq_len, [1, self.k_max])

        mask = self.sequence_mask(seq_len_tile, self.maxlen)
        pad = paddle.ones_like(mask, dtype="float32") * (-2 ** 32 + 1)
        # S*e
        low_capsule_new = paddle.matmul(item_his_emb,
                                        self.bilinear_mapping_matrix)
        if self.more_dropout:
            low_capsule_new = self.dropout(low_capsule_new)

        low_capsule_new_tile = paddle.tile(low_capsule_new, [1, 1, self.k_max])
        low_capsule_new_tile = paddle.reshape(
            low_capsule_new_tile,
            [-1, self.maxlen, self.k_max, self.output_units])
        low_capsule_new_tile = paddle.transpose(low_capsule_new_tile,
                                                [0, 2, 1, 3])
        low_capsule_new_tile = paddle.reshape(
            low_capsule_new_tile,
            [-1, self.k_max, self.maxlen, self.output_units])
        low_capsule_new_nograd = paddle.assign(low_capsule_new_tile)
        low_capsule_new_nograd.stop_gradient = True

        B = paddle.tile(self.routing_logits,
                        [paddle.shape(item_his_emb)[0], 1, 1])
        B.stop_gradient = True

        for i in range(self.iters - 1):
            B_mask = paddle.where(mask, B, pad)
            # print(B_mask)
            W = F.softmax(B_mask, axis=2)
            W = paddle.unsqueeze(W, axis=2)
            high_capsule_tmp = paddle.matmul(W, low_capsule_new_nograd)
            # print(low_capsule_new_nograd.shape)
            high_capsule = self.squash(high_capsule_tmp)
            B_delta = paddle.matmul(
                low_capsule_new_nograd,
                paddle.transpose(high_capsule, [0, 1, 3, 2]))
            B_delta = paddle.reshape(
                B_delta, shape=[-1, self.k_max, self.maxlen])
            if self.more_dropout:
                B_delta = self.dropout(B_delta)
            B += B_delta

        B_mask = paddle.where(mask, B, pad)
        W = F.softmax(B_mask, axis=1)
        W = paddle.unsqueeze(W, axis=2)
        interest_capsule = paddle.matmul(W, low_capsule_new_tile)
        interest_capsule = self.squash(interest_capsule)
        if self.more_dropout:
            interest_capsule = self.dropout(interest_capsule)
        high_capsule = paddle.reshape(interest_capsule,
                                      [-1, self.k_max, self.output_units])

        user_features_tile = paddle.tile(user_features_emb, [1, self.k_max])
        user_features_vec = paddle.reshape(user_features_tile, [-1, self.k_max, self.output_units])

        capsule_concat = paddle.concat([high_capsule, user_features_vec], axis=-1)

        high_capsule = F.relu(self.relu_layer(capsule_concat))
        return high_capsule, W, seq_len


class MindLayer(nn.Layer):
    """MindLayer
    """

    def __init__(self,
                 item_count,
                 country_count,
                 user_country_count,
                 ads_campaign_count,
                 ads_group_count,
                 brand_count,
                 height_count,
                 phone_model_count,
                 product_count,
                 embedding_dim,
                 hidden_size,
                 neg_samples=100,
                 maxlen=30,
                 pow_p=1.0,
                 capsual_iters=3,
                 capsual_max_k=3,
                 capsual_init_std=1.0,
                 dropout=0.2,
                 more_features=False,
                 more_dropout=False,
                 batch_size=None):
        super(MindLayer, self).__init__()
        self.pow_p = pow_p
        self.hidden_size = hidden_size
        self.item_count = item_count
        self.country_count = country_count
        self.user_country_count = user_country_count
        self.item_id_range = paddle.arange(end=item_count, dtype="int64")
        self.item_emb = nn.Embedding(
            item_count,
            embedding_dim,
            padding_idx=0,
            weight_attr=paddle.ParamAttr(
                name="item_emb",
                initializer=nn.initializer.XavierUniform(
                    fan_in=item_count, fan_out=embedding_dim)))
        # print(self.item_emb.weight)
        self.embedding_bias = self.create_parameter(
            shape=(item_count,),
            is_bias=True,
            attr=paddle.ParamAttr(
                name="embedding_bias", trainable=False),
            default_initializer=nn.initializer.Constant(0))

        self.country_emb = nn.Embedding(
            country_count,
            embedding_dim,
            padding_idx=0,
            weight_attr=paddle.ParamAttr(
                name="country_emb",
                initializer=nn.initializer.XavierUniform(
                    fan_in=country_count, fan_out=embedding_dim)))

        self.user_country_emb = nn.Embedding(
            user_country_count,
            embedding_dim,
            padding_idx=0,
            weight_attr=paddle.ParamAttr(
                name="user_country_emb",
                initializer=nn.initializer.XavierUniform(
                    fan_in=user_country_count, fan_out=embedding_dim)))

        self.ads_campaign_emb = nn.Embedding(
            ads_campaign_count,
            embedding_dim,
            padding_idx=0,
            weight_attr=paddle.ParamAttr(
                name="ads_campaign_emb",
                initializer=nn.initializer.XavierUniform(
                    fan_in=ads_campaign_count, fan_out=embedding_dim)))

        self.ads_group_emb = nn.Embedding(
            ads_group_count,
            embedding_dim,
            padding_idx=0,
            weight_attr=paddle.ParamAttr(
                name="ads_group_emb",
                initializer=nn.initializer.XavierUniform(
                    fan_in=ads_group_count, fan_out=embedding_dim)))

        self.brand_emb = nn.Embedding(
            brand_count,
            embedding_dim,
            padding_idx=0,
            weight_attr=paddle.ParamAttr(
                name="brand_emb",
                initializer=nn.initializer.XavierUniform(
                    fan_in=brand_count, fan_out=embedding_dim)))

        self.height_emb = nn.Embedding(
            height_count,
            embedding_dim,
            padding_idx=0,
            weight_attr=paddle.ParamAttr(
                name="height_emb",
                initializer=nn.initializer.XavierUniform(
                    fan_in=height_count, fan_out=embedding_dim)))

        self.phone_model_emb = nn.Embedding(
            phone_model_count,
            embedding_dim,
            padding_idx=0,
            weight_attr=paddle.ParamAttr(
                name="phone_model_emb",
                initializer=nn.initializer.XavierUniform(
                    fan_in=phone_model_count, fan_out=embedding_dim)))

        self.product_emb = nn.Embedding(
            product_count,
            embedding_dim,
            padding_idx=0,
            weight_attr=paddle.ParamAttr(
                name="product_emb",
                initializer=nn.initializer.XavierUniform(
                    fan_in=product_count, fan_out=embedding_dim)))

        self.capsual_layer = Mind_Capsual_Layer(
            embedding_dim,
            hidden_size,
            maxlen=maxlen,
            iters=capsual_iters,
            k_max=capsual_max_k,
            init_std=capsual_init_std,
            dropout=dropout,
            more_dropout=more_dropout,
            batch_size=batch_size)
        self.sampled_softmax = Mind_SampledSoftmaxLoss_Layer(
            item_count, neg_samples, batch_size=batch_size)

        self.dropout = nn.Dropout(p=dropout)
        self.more_features = more_features
        self.more_dropout = more_dropout
        self.item_emb_linear = nn.Linear(in_features=2 * embedding_dim, out_features=embedding_dim)
        self.user_country_emb_linear = nn.Linear(in_features=embedding_dim, out_features=hidden_size)
        self.user_features_emb_linear = nn.Linear(in_features=7 * embedding_dim, out_features=hidden_size)

        self.output_softmax_linear = nn.Linear(in_features=hidden_size, out_features=item_count)
        self.loss = nn.CrossEntropyLoss()

    def label_aware_attention(self, keys, query):
        """label_aware_attention
        """
        weight = paddle.matmul(keys,
                               paddle.reshape(query, [
                                   -1, paddle.shape(query)[-1], 1
                               ]))  # [B, K, dim] * [B, dim, 1] == [B, k, 1]
        weight = paddle.squeeze(weight, axis=-1)
        weight = paddle.pow(weight, self.pow_p)  # [x,k_max]
        weight = F.softmax(weight)  # [x, k_max]
        weight = paddle.unsqueeze(weight, 1)  # [B, 1, k_max]
        output = paddle.matmul(
            weight, keys)  # [B, 1, k_max] * [B, k_max, dim] => [B, 1, dim]
        return output.squeeze(1), weight

    def forward(self, hist_item, seqlen, labels=None, hist_country=None, user_country=None, item_country=None,
                ads_campaign=None, ads_group=None, brand=None, height=None, phone_model=None, product=None):
        """forward

        Args:
            hist_item : [B, maxlen, 1]
            seqlen : [B, 1]
            labels : [B, 1]
            hist_country : [B, maxlen, 1]
            user_country : [B, 1]
            item_country : [B, 1]
            ads_campaign : [B, 1]
            ads_group : [B, 1]
            brand : [B, 1]
            height : [B, 1]
            phone_model : [B, 1]
            product : [B, 1]
        """

        hit_item_emb = self.item_emb(hist_item)  # [B, seqlen, embed_dim]
        hit_country_emb = self.country_emb(hist_country)
        hit_item_emb = self.dropout(hit_item_emb)
        hit_country_emb = self.dropout(hit_country_emb)

        hit_concat_emb = paddle.concat([hit_item_emb, hit_country_emb], axis=-1)
        hist_emb = F.relu(self.item_emb_linear(hit_concat_emb))
        if self.more_dropout:
            hist_emb = self.dropout(hist_emb)

        user_country_emb = self.user_country_emb(user_country)
        user_country_emb = self.dropout(user_country_emb)

        if self.more_features:
            ads_campaign_emb = self.ads_campaign_emb(ads_campaign)
            ads_group_emb = self.ads_group_emb(ads_group)
            brand_emb = self.brand_emb(brand)
            height_emb = self.height_emb(height)
            phone_model_emb = self.phone_model_emb(phone_model)
            product_emb = self.product_emb(product)

            ads_campaign_emb = self.dropout(ads_campaign_emb)
            ads_group_emb = self.dropout(ads_group_emb)
            brand_emb = self.dropout(brand_emb)
            height_emb = self.dropout(height_emb)
            phone_model_emb = self.dropout(phone_model_emb)
            product_emb = self.dropout(product_emb)

            concat_emb = paddle.concat([user_country_emb, ads_campaign_emb, ads_group_emb, brand_emb, height_emb,
                                        phone_model_emb, product_emb],
                                       axis=-1)
            user_features_emb = F.relu(self.user_features_emb_linear(concat_emb))
        else:
            user_features_emb = F.relu(self.user_country_emb_linear(user_country_emb))

        if self.more_dropout:
            user_features_emb = self.dropout(user_features_emb)

        user_cap, cap_weights, cap_mask = self.capsual_layer(hist_emb,
                                                             seqlen,
                                                             user_features_emb)

        if not self.training:
            return user_cap, cap_weights

        target_emb = self.item_emb(labels)
        target_country_emb = self.country_emb(item_country)
        target_emb = self.dropout(target_emb)
        target_country_emb = self.dropout(target_country_emb)

        target_concat_emb = paddle.concat([target_emb, target_country_emb], axis=-1)
        target_emb = F.relu(self.item_emb_linear(target_concat_emb))

        user_emb, W = self.label_aware_attention(user_cap, target_emb)

        logits = self.output_softmax_linear(user_emb)

        loss = self.loss(logits, labels)

        return loss, W, user_cap, cap_weights, cap_mask
