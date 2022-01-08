# Copyright 2018 The Cornac Authors. All Rights Reserved.
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
# ============================================================================

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers, Input

from ...utils import get_rng
from ...utils.init_utils import uniform

def get_data(batch_ids, train_set, max_text_length, by='user', max_num_review=32):
    from tensorflow.python.keras.preprocessing.sequence import pad_sequences
    batch_reviews, batch_id_reviews, batch_num_reviews = [], [], []
    review_group = train_set.review_text.user_review if by == 'user' else train_set.review_text.item_review
    for idx in batch_ids:
        ids, review_ids = [], []
        for inc, (jdx, review_idx) in enumerate(review_group[idx].items()):
            if max_num_review is not None and inc == max_num_review:
                break
            ids.append(jdx)
            review_ids.append(review_idx)
        batch_id_reviews.append(ids)
        reviews = train_set.review_text.batch_seq(review_ids, max_length=max_text_length)
        batch_reviews.append(reviews)
        batch_num_reviews.append(len(reviews))
    batch_reviews = pad_sequences(batch_reviews, padding="post")
    batch_id_reviews = pad_sequences(batch_id_reviews, padding="post")
    batch_num_reviews = np.array(batch_num_reviews)
    return batch_reviews, batch_id_reviews, batch_num_reviews


class Model:
    def __init__(self, n_users, n_items, vocab, global_mean,
                 n_factors=32, embedding_size=100, id_embedding_size=32, attention_size=16, dropout_rate=0.5,
                 max_text_length=50, max_num_review=32, pretrained_word_embeddings=None, verbose=False, seed=None):
        self.n_users = n_users
        self.n_items = n_items
        self.n_vocab = vocab.size
        self.global_mean = global_mean
        self.n_factors = n_factors
        self.embedding_size = embedding_size
        self.id_embedding_size = id_embedding_size
        self.attention_size = attention_size
        self.dropout_rate = dropout_rate
        self.max_text_length = max_text_length
        self.max_num_review = max_num_review
        self.verbose = verbose
        if seed is not None:
            self.rng = get_rng(seed)
            tf.random.set_seed(seed)

        embedding_matrix = uniform(shape=(self.n_vocab, self.embedding_size), low=-0.5, high=0.5, random_state=self.rng)
        embedding_matrix[:4, :] = np.zeros((4, self.embedding_size))
        if pretrained_word_embeddings is not None:
            oov_count = 0
            for word, idx in vocab.tok2idx.items():
                embedding_vector = pretrained_word_embeddings.get(word)
                if embedding_vector is not None:
                    embedding_matrix[idx] = embedding_vector
                else:
                    oov_count += 1
            if self.verbose:
                print("Number of OOV words: %d" % oov_count)

        embedding_matrix = initializers.Constant(embedding_matrix)
        i_user_id = Input(shape=(1,), dtype="int32", name="input_user_id")
        i_item_id = Input(shape=(1,), dtype="int32", name="input_item_id")
        i_user_review = Input(shape=(None, self.max_text_length), dtype="int32", name="input_user_review")
        i_item_review = Input(shape=(None, self.max_text_length), dtype="int32", name="input_item_review")
        i_user_iid_review = Input(shape=(None,), dtype="int32", name="input_user_iid_review")
        i_item_uid_review = Input(shape=(None,), dtype="int32", name="input_item_uid_review")
        i_user_num_reviews = Input(shape=(1,), dtype="int32", name="input_user_number_of_review")
        i_item_num_reviews = Input(shape=(1,), dtype="int32", name="input_item_number_of_review")

        l_user_review_embedding = layers.Embedding(self.n_vocab, self.embedding_size, embeddings_initializer=embedding_matrix, mask_zero=True, name="layer_user_review_embedding")
        l_item_review_embedding = layers.Embedding(self.n_vocab, self.embedding_size, embeddings_initializer=embedding_matrix, mask_zero=True, name="layer_item_review_embedding")
        l_user_iid_embedding = layers.Embedding(self.n_items, self.id_embedding_size, embeddings_initializer="uniform", name="user_iid_embedding")
        l_item_uid_embedding = layers.Embedding(self.n_users, self.id_embedding_size, embeddings_initializer="uniform", name="item_uid_embedding")
        l_user_embedding = layers.Embedding(self.n_users, self.id_embedding_size, embeddings_initializer="uniform", name="user_embedding")
        l_item_embedding = layers.Embedding(self.n_items, self.id_embedding_size, embeddings_initializer="uniform", name="item_embedding")
        user_bias = layers.Embedding(self.n_users, 1, embeddings_initializer=tf.initializers.Constant(0.1), name="user_bias")
        item_bias = layers.Embedding(self.n_items, 1, embeddings_initializer=tf.initializers.Constant(0.1), name="item_bias")

        v_ui = layers.Multiply(name="vui")([l_user_embedding(i_user_id), l_item_embedding(i_item_id)])
        user_review_embedding = l_user_review_embedding(i_user_review)
        item_review_embedding = l_item_review_embedding(i_item_review)

        l_user_intra_review = layers.Bidirectional(
            layers.GRU(n_factors, return_sequences=True),
            merge_mode='sum'
        )
        l_item_intra_review = layers.Bidirectional(
            layers.GRU(n_factors, return_sequences=True),
            merge_mode='sum'
        )

        user_word_h = l_user_intra_review(
            tf.reshape(user_review_embedding, shape=[-1, self.max_text_length, self.embedding_size])
        )
        item_word_h = l_item_intra_review(
            tf.reshape(item_review_embedding, shape=[-1, self.max_text_length, self.embedding_size])
        )
        alpha_user = layers.Dense(1, activation=None, use_bias=True)(
            layers.Dense(self.attention_size, activation="relu", use_bias=True)(
                tf.concat([user_word_h, tf.tile(v_ui, [tf.shape(user_review_embedding)[1], self.max_text_length, 1])], axis=-1)
            )
        )

        alpha_item = layers.Dense(1, activation=None, use_bias=True)(
            layers.Dense(self.attention_size, activation="relu", use_bias=True)(
                tf.concat([item_word_h, tf.tile(v_ui, [tf.shape(item_review_embedding)[1], self.max_text_length, 1])], -1)
            )
        )
        alpha_user_attention = layers.Softmax(axis=1)(tf.reshape(alpha_user, [-1, self.max_text_length]), tf.reshape(user_review_embedding._keras_mask, shape=[-1, self.max_text_length]))
        alpha_item_attention = layers.Softmax(axis=1)(tf.reshape(alpha_item, [-1, self.max_text_length]), tf.reshape(item_review_embedding._keras_mask, shape=[-1, self.max_text_length]))
        user_word_h_ = tf.math.reduce_mean(tf.multiply(tf.expand_dims(alpha_user_attention, -1), user_word_h), 1)
        item_word_h_ = tf.math.reduce_mean(tf.multiply(tf.expand_dims(alpha_item_attention, -1), item_word_h), 1)

        user_inter_review_gru = layers.Bidirectional(
            layers.GRU(self.n_factors, return_sequences=True),
            merge_mode='sum'
        )
        item_inter_review_gru = layers.Bidirectional(
            layers.GRU(self.n_factors, return_sequences=True),
            merge_mode='sum'
        )
        user_review_h = user_inter_review_gru(
            tf.reshape(user_word_h_, shape=[-1, tf.shape(user_review_embedding)[1], self.n_factors])
        )
        item_review_h = item_inter_review_gru(
            tf.reshape(item_word_h_, shape=[-1, tf.shape(item_review_embedding)[1], self.n_factors])
        )
        a_user = layers.Dense(1, activation=None, use_bias=True)(
            layers.Dense(self.attention_size, activation="relu", use_bias=True)(
                tf.concat([tf.multiply(user_review_h, l_user_iid_embedding(i_user_iid_review)), tf.tile(v_ui, [1, tf.shape(user_review_embedding)[1], 1])], axis=-1)
            )
        )

        a_user_masking = tf.expand_dims(tf.sequence_mask(tf.reshape(i_user_num_reviews, [-1]), maxlen=tf.shape(user_review_embedding)[1]), -1)
        user_attention = layers.Softmax(axis=1)(a_user, a_user_masking)

        a_item = layers.Dense(1, activation=None, use_bias=True)(
            layers.Dense(self.attention_size, activation="relu", use_bias=True)(
                tf.concat([tf.multiply(item_review_h, l_item_uid_embedding(i_item_uid_review)), tf.tile(v_ui, [1, tf.shape(item_review_embedding)[1], 1])], axis=-1)
            )
        )

        a_item_masking = tf.expand_dims(tf.sequence_mask(tf.reshape(i_item_num_reviews, [-1]), maxlen=tf.shape(item_review_embedding)[1]), -1)
        item_attention = layers.Softmax(axis=1)(a_item, a_item_masking)
        c = layers.Dense(n_factors, activation=None, use_bias=True, name="Wc")(
            tf.concat([
                tf.reduce_sum(layers.Multiply()([user_attention, user_review_h, l_item_uid_embedding(i_item_uid_review)]), 1, name="su"),
                tf.reduce_sum(layers.Multiply()([item_attention, item_review_h, l_item_uid_embedding(i_item_uid_review)]), 1, name="si"),
                tf.reduce_sum(v_ui, 1)
            ], axis=1)
        )

        W1 = layers.Dense(1, activation=None, use_bias=False, name="W1")
        r = layers.Add(name="prediction")([
            W1(c),
            user_bias(i_user_id),
            item_bias(i_item_id),
            keras.backend.constant(self.global_mean, shape=(1,), name="global_mean"),
        ])

        self.graph = keras.Model(inputs=[i_user_id, i_item_id, i_user_review, i_user_iid_review, i_user_num_reviews, i_item_review, i_item_uid_review, i_item_num_reviews], outputs=r)
        if self.verbose:
            self.graph.summary()

    def get_weights(self, train_set, batch_size=64):
        user_attention_review = keras.Model(inputs=[self.graph.get_layer('input_user_review').input, self.graph.get_layer('input_user_iid_review').input, self.graph.get_layer('input_user_number_of_review').input], outputs=self.graph.get_layer('su').output)
        item_attention_review = keras.Model(inputs=[self.graph.get_layer('input_item_review').input, self.graph.get_layer('input_item_uid_review').input, self.graph.get_layer('input_item_number_of_review').input], outputs=self.graph.get_layer('si').output)
        Su = np.zeros((self.n_users, self.n_factors))
        Si = np.zeros((self.n_items, self.n_factors))
        for batch_users in train_set.user_iter(batch_size):
            user_reviews, user_iid_reviews, user_num_reviews = get_data(batch_users, train_set, self.max_text_length, by='user')
            Su_ = user_attention_review([user_reviews, user_iid_reviews, user_num_reviews], training=False)
            Su[batch_users] = Su_.numpy()
        for batch_items in train_set.item_iter(batch_size):
            item_reviews, item_uid_reviews, item_num_reviews = get_data(batch_items, train_set, self.max_text_length, by='item')
            Si_ = item_attention_review([item_reviews, item_uid_reviews, item_num_reviews], training=False)
            Si[batch_items] = Si_.numpy()
        Wc = self.graph.get_layer('Wc').get_weights()[0]
        bc = self.graph.get_layer('Wc').get_weights()[1]
        W1 = self.graph.get_layer('W1').get_weights()[0]
        user_embedding = self.graph.get_layer('user_embedding').get_weights()[0]
        item_embedding = self.graph.get_layer('item_embedding').get_weights()[0]
        bu = self.graph.get_layer('user_bias').get_weights()[0]
        bi = self.graph.get_layer('item_bias').get_weights()[0]
        mu = self.global_mean
        return Su, Si, Wc, bc, W1, user_embedding, item_embedding, bu, bi, mu
