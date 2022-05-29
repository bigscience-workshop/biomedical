# %%
from pathlib import Path
import datasets
import transformers
import functools
import torch
from collections import defaultdict
from tqdm.auto import tqdm
# %%
def tokenize(example, tokenizer):
    tokenized = tokenizer(example["tokens"], is_split_into_words=True)
    tokenized["word_ids"] = tokenized.word_ids()
    return tokenized

# %%
if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model", required=True)
    # parser.add_argument("--data", required=True, type=Path)
    # args = parser.parse_args()

    # %%
    model_name = "leonweber/foo"
    dataset_path = Path("/vol/fob-vol1/mi15/weberple/projects/LinkBERT/data/tokcls/NCBI-disease_hf_machamp")
    data_files = {"dev": str(dataset_path / "dev.json")}
    dataset = datasets.load_dataset("json", data_files=data_files, split="dev")

    head = "ncbi"


    # %%
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = transformers.AutoModelForTokenClassification.from_pretrained(model_name).cuda()

    # %%
    dataset_tokenized = dataset.map(functools.partial(tokenize, tokenizer=tokenizer), remove_columns=dataset.column_names)
    all_word_ids = dataset_tokenized["word_ids"]
    dataset_tokenized = dataset_tokenized.remove_columns(["word_ids"])

    # %%
    dataloader_dev = torch.utils.data.DataLoader(dataset_tokenized, batch_size=32, shuffle=False,
        collate_fn=transformers.DataCollatorWithPadding(tokenizer=tokenizer))
    
    all_predictions = []
    idx_example = 0
    for batch in tqdm(dataloader_dev):
        batch_logits = model(**batch.to("cuda"))["logits"]
        for logits, mask in zip(batch_logits, batch["attention_mask"]):
            word_ids = all_word_ids[idx_example]
            predictions = []
            valid_logits = logits[mask == 1]
            prev_word_id = None
            for token_logits, word_id in zip(valid_logits, word_ids):
                if word_id is None or word_id == prev_word_id:
                    continue
                predictions.append({model.config.id2label[i]: l.item() for i, l in enumerate(token_logits) if head in model.config.id2label[i].lower()})
                prev_word_id = word_id
            all_predictions.append(predictions)
            idx_example += 1

    dataset = dataset.add_column("predictions", all_predictions)
    out_file = dataset_path / model_name / "predictions_dev.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_json(out_file)
    
