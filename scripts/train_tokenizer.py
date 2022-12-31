import multiprocessing

import datasets
from loguru import logger
import numpy as np
from transformers import AutoTokenizer


NUM_PROC = multiprocessing.cpu_count()


def get_training_corpus(dataset, batch_size=1_000):
    for start_idx in range(0, len(dataset), batch_size):
        samples = dataset[start_idx : start_idx + batch_size]
        yield samples["text"]

        
meta_ds_base_name = "bigbio-public-text-concat"
clone_from_name = "gpt2"
batch_size = 1_000
vocab_size = 25_000

dsd = {}
for split_name in ["train", "validation", "test"]:
    meta_ds_name = f"{meta_ds_base_name}-{split_name}"
    logger.info(f"reading {meta_ds_name}")
    dsd[split_name] = datasets.load_from_disk(meta_ds_name)

ds_train = dsd["train"]
clone_from_tokenizer = AutoTokenizer.from_pretrained(clone_from_name)
training_corpus = get_training_corpus(ds_train, batch_size)

# TODO: check why this is much faster when you save individual dataset
# to disk as opposed to dataset dictionary.
tokenizer = clone_from_tokenizer.train_new_from_iterator(training_corpus, vocab_size)
tokenizer.save_pretrained("bigbio-public-gpt2-v25k-tokenizer")
