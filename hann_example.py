import os
import cornac
from cornac.data import Reader
from cornac.datasets import amazon_digital_music
from cornac.eval_methods import RatioSplit
from cornac.data import ReviewModality
from cornac.data.text import BaseTokenizer
import pandas as pd
import numpy as np

import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

k = 8
emb_size = 100
BATCH_SIZE = 64
# BATCH_SIZE = 1
MIN_ITEM_FREQ = 1
MAX_TEXT_LENGTH = 50
MAX_NUM_REVIEW = 10
data_dir = "experiments/eqa/dist/toy/"
# data_dir = 'experiments/eqa/dist/electronic'
# data_dir = 'experiments/eqa/dist/video_game'
# data_dir = 'experiments/eqa/dist/automotive'
# data_dir = 'experiments/eqa/dist/cellphone'
# data_dir = 'experiments/eqa/dist/patio'
# data_dir = 'experiments/eqa/dist/beauty'
# data_dir = 'experiments/eqa/dist/baby'
# feedback = amazon_digital_music.load_feedback()
feedback = Reader(min_item_freq=MIN_ITEM_FREQ).read(
    os.path.join(data_dir, "rating.txt"), fmt="UIR", sep="\t"
)
# reviews = amazon_digital_music.load_review()
reviews = Reader().read(os.path.join(data_dir, "review.txt"), fmt="UIReview", sep="\t")

pretrained_word_embeddings = {}
with open(
    "/data/shared/download/glove/glove.6B.{}d.txt".format(emb_size), encoding="utf-8"
) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        pretrained_word_embeddings[word] = coefs

review_modality = ReviewModality(
    data=reviews,
    tokenizer=BaseTokenizer(stop_words="english"),
    max_vocab=4000,
    max_doc_freq=0.5,
)

ratio_split = RatioSplit(
    data=feedback,
    # test_size=0.1,
    # val_size=0.1,
    test_size=0.4,
    val_size=0.4,
    exclude_unknowns=True,
    review_text=review_modality,
    verbose=True,
    seed=123,
)

# Put everything together into an experiment and run it
cornac.Experiment(
    eval_method=ratio_split,
    models=[
        cornac.models.HANN(
            name=f"HANN_MINITEM_{MIN_ITEM_FREQ}_MAXNREVIEW_{MAX_NUM_REVIEW}_EMB_{100}_IDEMB_{k}_K_{k}_ATT_{k}_MAXTXTLEN_{MAX_TEXT_LENGTH}_BS_{BATCH_SIZE}_E_{max_iter}",
            embedding_size=100,
            n_factors=k,
            id_embedding_size=k,
            attention_size=k,
            batch_size=BATCH_SIZE,
            max_text_length=MAX_TEXT_LENGTH,
            # max_num_review=MAX_NUM_REVIEW,
            dropout_rate=0.5,
            max_iter=max_iter,
            init_params={"pretrained_word_embeddings": pretrained_word_embeddings},
            seed=123,
        )
        for k in [8]
        # for max_iter in [10,9,8,7,6,5,4,3,2,1]
        # for max_iter in [40, 30, 20, 15]
        for max_iter in [1]
    ],
    metrics=[cornac.metrics.RMSE()],
).run()

print(data_dir)