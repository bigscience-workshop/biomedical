# %%
import json
import datasets
from seqeval.metrics import f1_score, classification_report, accuracy_score
import re


# %%
data = datasets.load_dataset("json", data_files="/vol/fob-vol1/mi15/weberple/projects/LinkBERT/data/tokcls/NCBI-disease_hf/leonweber/foo/predictions_dev.json")

# %%
y_true = []
y_pred = []

tag_regex = r".*"
for i, example in enumerate(data["train"]):
    y_true.append(example["ner_tags"])
    preds = []

    for prediction in example["predictions"]:
        max_tag = None
        max_score = -float("inf")
        for tag, score in prediction.items():
            if re.match(tag_regex, tag) and score > max_score:
                max_tag = tag
                max_score = score
        preds.append(max_tag.split(":")[-1][0])
    y_pred.append(preds)


# %%
print(accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred))
# %%
