import json
from collections import Counter

from rich import print as rprint


def load_dataset_metadata(filename):
    with open(filename, "r") as metadata:
        m = json.load(metadata)
    return m


def count_split_by_task_dataset(dataset_metadata, count_type, counter):
    """
    counter is in this format:
        "<dataset_schema_name_split>: num_count_type"
    """
    configs = dataset_metadata["config_metas"]
    print(configs.keys())
    agg_schema = {}
    for key, schema in configs.items():
        bigbio_type = schema['bigbio_schema']
        if bigbio_type not in agg_schema.keys():
            agg_schema[bigbio_type] = []
        agg_schema[bigbio_type].append(key)
    print(agg_schema)
    for a, vv in agg_schema.items():
        for sub_schema in vv:
            split_data = configs[sub_schema]["splits"]
            for s, v in split_data.items():
                try:
                    sample_count = v[count_type]
                    sample_count_key = "_".join([sub_schema, s])
                    counter[sample_count_key] += sample_count
                except KeyError:
                    print("key not found skipping")
                    continue
    return counter


if __name__ == "__main__":
    meta_file_name = "dataset_metadatas.json"

    dataset_name = "ask_a_patient"
    meta = load_dataset_metadata(meta_file_name)
    agg_keys = meta.keys()
    entry = meta[dataset_name]
    entry_keys = entry.keys()
    count_type = ["samples_count", "entities_count", "events_count", "relations_count"]
    count_data = {}
    # num_sample_by_split = Counter()
    # c = count_split_by_task_dataset(entry, 'samples_count', num_sample_by_split)
    # print(c)
    for ct in count_type:
        num_sample_by_split = Counter()
        for dataset_name, entry in meta.items():
            c = count_split_by_task_dataset(entry, ct, num_sample_by_split)
        count_data[ct] = c

    with open("dataset_by_task_by_split.json", "w") as fo:
        json.dump(count_data, fo, indent=4)
