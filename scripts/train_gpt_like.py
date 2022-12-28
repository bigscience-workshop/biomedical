import multiprocessing

import datasets
from loguru import logger
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import Trainer
from transformers import TrainingArguments


NUM_PROC = multiprocessing.cpu_count()


def get_training_corpus(dataset, batch_size=1_000):
    for start_idx in range(0, len(dataset), batch_size):
        samples = dataset[start_idx : start_idx + batch_size]
        yield samples["text"]

def map_tokenize(examples):
    return tokenizer(examples['text'])

def map_batch_num_tokens(examples):
    return {"num_tokens": [len(el) for el in examples["input_ids"]]}


meta_ds_name = "bigbio-public-text-concat"
ds_all = datasets.load_from_disk(meta_ds_name)
ds_all = ds_all.shuffle(seed=42)
ds_sml = ds_all.select(np.arange(500_000))

ds_text = ds_sml
#ds_train = ds_all



model_checkpt = "gpt2"
tokenizer_checkpt = "bigbio-public-gpt2-v20k-tokenizer"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpt)

batch_size = 1_000
training_corpus = get_training_corpus(ds_text, batch_size)
ds_tokenized = ds_text.map(
    map_tokenize,
    batched=True,
    num_proc=NUM_PROC,
    remove_columns=["text"],
)



# block_size = tokenizer.model_max_length
block_size = 128


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model
    # supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


ds_lm = ds_tokenized.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=NUM_PROC,
)


config = AutoConfig.from_pretrained(model_checkpt)
model = AutoModelForCausalLM.from_config(config)

training_args = TrainingArguments(
    f"bigbio-{model_checkpt}",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
)

trainer.train()
