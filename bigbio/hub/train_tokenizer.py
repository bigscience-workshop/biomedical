import datasets
import numpy as np
from transformers import AutoTokenizer


NUM_PROC = 8


def get_training_corpus(dataset, batch_size=1_000):
    for start_idx in range(0, len(dataset), batch_size):
        samples = dataset[start_idx : start_idx + batch_size]
        yield samples["text"]

def map_tokenize(examples):
    return tokenizer(examples['text'])

def map_batch_num_tokens(examples):
    return {"num_tokens": [len(el) for el in examples["input_ids"]]}


meta_ds_name = "bigbio_public_text_concat"
clone_from_name = "gpt2"
batch_size = 1_000
vocab_size = 20_000


ds_all = datasets.load_from_disk(meta_ds_name)
ds_all = ds_all.shuffle(seed=42)
ds_sml = ds_all.select(np.arange(500_000))
#ds_train = ds_sml
ds_train = ds_all


clone_from_tokenizer = AutoTokenizer.from_pretrained(clone_from_name)

training_corpus = get_training_corpus(ds_train, batch_size)
tokenizer = clone_from_tokenizer.train_new_from_iterator(training_corpus, vocab_size)

ds_train = ds_train.map(map_tokenize, batched=True, num_proc=NUM_PROC)
ds_train = ds_train.map(map_batch_num_tokens, batched=True, num_proc=NUM_PROC)

total_tokens = sum(ds_train["num_tokens"])

logger.info("ds_train has {} million tokens.".format(total_tokens/1e6))
