import multiprocessing

import datasets
from loguru import logger
import numpy as np
from transformers import AutoTokenizer


NUM_PROC = multiprocessing.cpu_count()


def map_tokenize(examples):
    texts = examples["text"]
    assert isinstance(texts, list)
    assert all([isinstance(text, str) for text in texts])
    return tokenizer(examples["text"])


def map_batch_num_tokens(examples):
    return {"num_tokens": [len(el) for el in examples["input_ids"]]}


meta_ds_base_name = "bigbio-public-text-concat"
clone_from_name = "gpt2"
batch_size = 1_000


dsd = {}
for split_name in ["train", "validation", "test"]:
    meta_ds_name = f"{meta_ds_base_name}-{split_name}"
    logger.info(f"reading {meta_ds_name}")
    dsd[split_name] = datasets.load_from_disk(meta_ds_name)

tokenizer = AutoTokenizer.from_pretrained("bigbio-public-gpt2-v25k-tokenizer")


# this tokenizes all the datasets  just to count the tokens
# shuffling helps keep the load balanced
for split_name in ["train", "validation", "test"]:
    logger.info(f"tokenizing {split_name}")
    ds = dsd[split_name].shuffle(seed=42)
    #ds = dsd[split_name]
    #ds = ds.map(map_tokenize)
    ds = ds.map(map_tokenize, batched=True, num_proc=NUM_PROC, batch_size=1000)
    ds = ds.map(map_batch_num_tokens, batched=True, num_proc=NUM_PROC)

    total_tokens = sum(ds["num_tokens"])
    logger.info("ds has {} million tokens.".format(total_tokens / 1e6))
