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

import os
import numpy as np
from tqdm.auto import trange

from ..recommender import Recommender
from ...exception import ScoreException


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class HANN(Recommender):
    """Hierarchical Attention based Neural Network

    Parameters
    ----------
    name: string, default: 'HANN'
        The name of the recommender model.

    embedding_size: int, default: 100
        Word embedding size

    id_embedding_size: int, default: 32
        User/item review id embedding size

    n_factors: int, default: 32
        The dimension of the user/item's latent factors.

    attention_size: int, default: 16
        Attention size

    dropout_rate: float, default: 0.5
        Dropout rate of neural network dense layers

    max_text_length: int, default: 50
        Maximum number of tokens in a review instance

    batch_size: int, default: 64
        Batch size

    max_iter: int, default: 10
        Max number of training epochs

    trainable: boolean, optional, default: True
        When False, the model will not be re-trained, and input of pre-trained parameters are required.

    verbose: boolean, optional, default: True
        When True, running logs are displayed.

    init_params: dictionary, optional, default: None
        Initial parameters, pretrained_word_embeddings could be initialized here, e.g., init_params={'pretrained_word_embeddings': pretrained_word_embeddings}

    seed: int, optional, default: None
        Random seed for weight initialization.
        If specified, training will take longer because of single-thread (no parallelization).

    References
    ----------
    * Dawei Cong, Yanyan Zhao, Bing Qin, Yu Han, Murray Zhang, Alden Liu, and Nat Chen. 2019. Hierarchical Attention based Neural Network for Explainable Recommendation. In Proceedings of the 2019 on International Conference on Multimedia Retrieval (ICMR '19). Association for Computing Machinery, New York, NY, USA, 373–381. DOI:https://doi-org.libproxy.smu.edu.sg/10.1145/3323873.3326592
    """

    def __init__(
        self,
        name="HANN",
        embedding_size=100,
        id_embedding_size=32,
        n_factors=32,
        attention_size=16,
        dropout_rate=0.5,
        max_text_length=50,
        max_num_review=32,
        batch_size=64,
        max_iter=10,
        optimizer='adam',
        learning_rate=0.001,
        model_selection='last', # last or best
        user_based=True,
        trainable=True,
        verbose=True,
        init_params=None,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.seed = seed
        self.embedding_size = embedding_size
        self.id_embedding_size = id_embedding_size
        self.n_factors = n_factors
        self.attention_size = attention_size
        self.dropout_rate = dropout_rate
        self.max_text_length = max_text_length
        self.max_num_review = max_num_review
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.model_selection = model_selection
        self.user_based = user_based
        # Init params if provided
        self.init_params = {} if init_params is None else init_params
        self.losses = {"train_losses": [], "val_losses": []}

    def fit(self, train_set, val_set=None):
        """Fit the model to observations.

        Parameters
        ----------
        train_set: :obj:`cornac.data.Dataset`, required
            User-Item preference data as well as additional modalities.

        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            User-Item preference data for model selection purposes (e.g., early stopping).

        Returns
        -------
        self : object
        """
        Recommender.fit(self, train_set, val_set)

        if self.trainable:
            if not hasattr(self, "model"):
                from .hann import Model
                self.model = Model(
                    self.train_set.num_users,
                    self.train_set.num_items,
                    self.train_set.review_text.vocab,
                    self.train_set.global_mean,
                    n_factors=self.n_factors,
                    embedding_size=self.embedding_size,
                    id_embedding_size=self.id_embedding_size,
                    attention_size=self.attention_size,
                    dropout_rate=self.dropout_rate,
                    max_text_length=self.max_text_length,
                    max_num_review=self.max_num_review,
                    pretrained_word_embeddings=self.init_params.get('pretrained_word_embeddings'),
                    verbose=self.verbose,
                    seed=self.seed,
                )
            self._fit_hann()

        return self

    def _fit_hann(self):
        import tensorflow as tf
        from tensorflow import keras
        from .hann import get_data
        from ...eval_methods.base_method import rating_eval
        from ...metrics import MSE
        loss = keras.losses.MeanSquaredError()
        if not hasattr(self, 'optimizer_'):
            if self.optimizer == 'rmsprop':
                self.optimizer_ = keras.optimizers.RMSprop(learning_rate=self.learning_rate)
            else:
                self.optimizer_ = keras.optimizers.Adam(learning_rate=self.learning_rate)
        train_loss = keras.metrics.Mean(name="loss")
        val_loss = float('inf')
        best_val_loss = float('inf')
        loop = trange(self.max_iter, disable=not self.verbose)
        for i_epoch, _ in enumerate(loop):
            train_loss.reset_states()
            for i, (batch_users, batch_items, batch_ratings) in enumerate(self.train_set.uir_iter(self.batch_size, shuffle=True)):
                user_reviews, user_iid_reviews, user_num_reviews = get_data(batch_users, self.train_set, self.max_text_length, by='user', max_num_review=self.max_num_review)
                item_reviews, item_uid_reviews, item_num_reviews = get_data(batch_items, self.train_set, self.max_text_length, by='item', max_num_review=self.max_num_review)
                with tf.GradientTape() as tape:
                    predictions = self.model.graph(
                        [batch_users, batch_items, user_reviews, user_iid_reviews, user_num_reviews, item_reviews, item_uid_reviews, item_num_reviews],
                        training=True,
                    )
                    _loss = loss(predictions, batch_ratings)
                gradients = tape.gradient(_loss, self.model.graph.trainable_variables)
                self.optimizer_.apply_gradients(zip(gradients, self.model.graph.trainable_variables))
                train_loss(_loss)
                if i % 10 == 0:
                    loop.set_postfix(loss=train_loss.result().numpy())
            current_weights = self.model.get_weights(self.train_set, self.batch_size)
            if self.val_set is not None:
                self.Su, self.Si, self.Wc, self.bc, self.W1, self.user_embedding, self.item_embedding, self.bu, self.bi, self.mu = current_weights
                [current_val_mse], _ = rating_eval(
                    model=self,
                    metrics=[MSE()],
                    test_set=self.val_set,
                    user_based=self.user_based
                )
                val_loss = current_val_mse
                if best_val_loss > val_loss:
                    best_val_loss = val_loss
                    self.best_epoch = i_epoch + 1
                    best_weights = current_weights
                loop.set_postfix(loss=train_loss.result().numpy(), val_loss=val_loss, best_val_loss=best_val_loss, best_epoch=self.best_epoch)
            self.losses["train_losses"].append(train_loss.result().numpy())
            self.losses["val_losses"].append(val_loss)
        loop.close()

        # save weights for predictions
        self.Su, self.Si, self.Wc, self.bc, self.W1, self.user_embedding, self.item_embedding, self.bu, self.bi, self.mu = best_weights if self.val_set is not None and self.model_selection == 'best' else current_weights
        if self.verbose:
            print("Learning completed!")


    def save(self, save_dir=None):
        """Save a recommender model to the filesystem.

        Parameters
        ----------
        save_dir: str, default: None
            Path to a directory for the model to be stored.

        """
        if save_dir is None:
            return
        model = self.model
        del self.model

        model_file = Recommender.save(self, save_dir)

        self.model = model
        self.model.save(model_file.replace(".pkl", ".cpt"))

        return model_file

    @staticmethod
    def load(model_path, trainable=False):
        """Load a recommender model from the filesystem.

        Parameters
        ----------
        model_path: str, required
            Path to a file or directory where the model is stored. If a directory is
            provided, the latest model will be loaded.

        trainable: boolean, optional, default: False
            Set it to True if you would like to finetune the model. By default, 
            the model parameters are assumed to be fixed after being loaded.
        
        Returns
        -------
        self : object
        """
        from tensorflow import keras
        model = Recommender.load(model_path, trainable)
        model.model = keras.models.load_model(model.load_from.replace(".pkl", ".cpt"))

        return model

    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform score prediction.

        item_idx: int, optional, default: None
            The index of the item for that to perform score prediction.
            If None, scores for all known items will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items
        """
        if item_idx is None:
            if self.train_set.is_unk_user(user_idx):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d)" % user_idx
                )
            c = self.Wc * np.concatenate((np.broadcast_to(self.Su[user_idx], shape=(self.Si.shape[0], self.Su[user_idx].shape[0])), self.Si, self.user_embedding[user_idx] * self.item_embedding), axis=1) + self.bc
            known_item_scores = c.dot(self.W1) + self.bu[user_idx] + self.bi + self.mu
            return known_item_scores.ravel()
        else:
            if self.train_set.is_unk_user(user_idx) or self.train_set.is_unk_item(
                item_idx
            ):
                raise ScoreException(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )
            c = self.Wc * np.concatenate((self.Su[user_idx], self.Si[item_idx], self.user_embedding[user_idx] * self.item_embedding[item_idx]), axis=1) + self.bc
            known_item_score = c.dot(self.W1) + self.bu[user_idx] + self.bi[item_idx] + self.mu
            return known_item_score
