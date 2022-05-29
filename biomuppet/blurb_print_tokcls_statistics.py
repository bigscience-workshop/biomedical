import datasets
from collections import Counter

data = datasets.load_dataset("json", data_files="/vol/fob-vol1/mi15/weberple/projects/LinkBERT/data/tokcls/NCBI-disease_hf/leonweber/foo/predictions_dev.json")
counter = Counter()

for example in data["train"]:
    counter.update(example["tokens"])
    counter.update(example["ner_tags"])
    counter.update(["example"])

print(counter.most_common(10))