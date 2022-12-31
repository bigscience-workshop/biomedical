import multiprocessing

import datasets
from loguru import logger
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import Trainer
from transformers import TrainingArguments
import wandb


NUM_PROC = multiprocessing.cpu_count()


meta_ds_base_name = "bigbio-public-text-concat"
dsd = {}
for split_name in ["train", "validation", "test"]:
    meta_ds_name = f"{meta_ds_base_name}-{split_name}"
    logger.info(f"reading {meta_ds_name}")
    dsd[split_name] = datasets.load_from_disk(meta_ds_name)
dsd = datasets.DatasetDict(dsd)


model_checkpt = "gpt2"
tokenizer_checkpt = "bigbio-public-gpt2-v25k-tokenizer"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpt)

def map_tokenize(examples):
    return tokenizer(examples['text'])

dsd_tokenized = dsd.map(
    map_tokenize,
    batched=True,
    num_proc=NUM_PROC,
    remove_columns=["text"],
)

# block_size = tokenizer.model_max_length
block_size = 1024


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


dsd_lm = dsd_tokenized.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=NUM_PROC,
)


# see Table A4 in https://arxiv.org/abs/2203.15556
# https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2Config
CONFIG_PARAMS = {
    "73M": {
        "n_layer": 10,
        "n_embd": 640,
        "n_inner": 2560,
        "n_head": 10,
    },
    "305M": {
        "n_layer": 20,
        "n_embd": 1024,
        "n_inner": 4096,
        "n_head": 16,
    },
    "552M": {
        "n_layer": 24,
        "n_embd": 1280,
        "n_inner": 5120,
        "n_head": 10,
    },
}

config = AutoConfig.from_pretrained(
    model_checkpt,
    vocab_size=tokenizer.vocab_size,
    **CONFIG_PARAMS["73M"],
)
model = AutoModelForCausalLM.from_config(config)


wandb.init(project="bigbio-gpt")

training_args = TrainingArguments(
    f"bigbio-{model_checkpt}",
    logging_steps = 100,
    save_steps = 1_000,
    evaluation_strategy = "steps",
    do_eval = True,
    eval_steps = 1_000,
    per_device_eval_batch_size=64,
    do_train = True,
    num_train_epochs=2.0,
    gradient_accumulation_steps=4,
    per_device_train_batch_size=16,
    fp16=True,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.02,
    learning_rate=6e-4,
    weight_decay=0.01,
    report_to="wandb",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dsd_lm["train"],
    # TODO: probably want to take a sub-selection of validation ds here (slows down training)
    eval_dataset=dsd_lm["validation"].filter(lambda example, idx: idx % 32 == 0, with_indices=True),
)

trainer.train()
