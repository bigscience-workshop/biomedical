"""
Gather dataset statistics
Help make plots
"""
from collections import Counter
from collections import defaultdict, OrderedDict
import json
import os
from typing import Optional

import pandas as pd

import bigbio
from bigbio.dataloader import BigBioConfigHelpers


MAX_COMMON = 50


def get_kb_meta(helper, split, ds):

    passages_count = 0
    passages_char_count = 0
    passages_type_counter = Counter()

    entities_count = 0
    entities_type_counter = Counter()
    entities_db_name_counter = Counter()
    entities_unique_db_ids = set()

    events_count = 0
    events_type_counter = Counter()
    events_arguments_count = 0
    events_arguments_role_counter = Counter()

    coreferences_count = 0

    relations_count = 0
    relations_type_counter = Counter()
    relations_db_name_counter = Counter()
    relations_unique_db_ids = set()

    for sample in ds:
        for passage in sample["passages"]:
            passages_count += 1
            passages_char_count += len(passage["text"][0])
            passages_type_counter[passage["type"]] += 1

        for entity in sample["entities"]:
            entities_count += 1
            entities_type_counter[entity["type"]] += 1
            for norm in entity["normalized"]:
                entities_db_name_counter[norm["db_name"]] += 1
                entities_unique_db_ids.add(norm["db_id"])

        for event in sample["events"]:
            events_count += 1
            events_type_counter[event["type"]] += 1
            for argument in event["arguments"]:
                events_arguments_count += 1
                events_arguments_role_counter[argument["role"]] += 1

        for coreference in sample["coreferences"]:
            coreferences_count += 1

        for relation in sample["relations"]:
            relations_count += 1
            relations_type_counter[relation["type"]] += 1
            for norm in relation["normalized"]:
                relations_db_name_counter[norm["db_name"]] += 1
                relations_unique_db_ids.add(norm["db_id"])

    for cc in [
            passages_type_counter,
            entities_type_counter,
            entities_db_name_counter,
            events_arguments_role_counter,
            relations_type_counter
    ]:
        if None in cc.keys():
            raise ValueError()


    meta = {
        "split": split,
        "samples_count": ds.num_rows,
        "passages_count": passages_count,
        "passages_type_counts": dict(passages_type_counter.most_common(MAX_COMMON)),
        "passages_char_count": passages_char_count,
        "entities_count": entities_count,
        "entities_type_counts": dict(entities_type_counter.most_common(MAX_COMMON)),
        "entities_db_name_counts": dict(
            entities_db_name_counter.most_common(MAX_COMMON)
        ),
        "entities_unique_db_id_counts": len(entities_unique_db_ids),
        "events_count": events_count,
        "events_arguments_count": events_arguments_count,
        "events_arguments_role_counts": dict(
            events_arguments_role_counter.most_common(MAX_COMMON)
        ),
        "coreferences_count": coreferences_count,
        "relations_count": relations_count,
        "relations_type_counts": dict(relations_type_counter.most_common(MAX_COMMON)),
        "relations_db_name_counts": dict(
            relations_db_name_counter.most_common(MAX_COMMON)
        ),
        "relations_unique_db_id_counts": len(relations_unique_db_ids),
    }

    return meta


def get_text_meta(helper, split, ds):

    text_char_count = 0
    labels_count = 0
    labels_counter = Counter()

    for sample in ds:
        text_char_count += len(sample["text"]) if sample["text"] is not None else 0
        for label in sample["labels"]:
            labels_count += 1
            labels_counter[label] += 1

    meta = {
        "split": split,
        "samples_count": ds.num_rows,
        "text_char_count": text_char_count,
        "labels_count": labels_count,
        "labels_counts": dict(labels_counter.most_common(MAX_COMMON)),
    }

    return meta


def get_pairs_meta(helper, split, ds):

    text_1_char_count = 0
    text_2_char_count = 0
    label_counter = Counter()

    for sample in ds:
        text_1_char_count += (
            len(sample["text_1"]) if sample["text_1"] is not None else 0
        )
        text_2_char_count += (
            len(sample["text_2"]) if sample["text_2"] is not None else 0
        )
        label_counter[sample["label"]] += 1

    meta = {
        "split": split,
        "samples_count": ds.num_rows,
        "text_1_char_count": text_1_char_count,
        "text_2_char_count": text_2_char_count,
        "label_counts": dict(label_counter.most_common(MAX_COMMON)),
    }

    return meta


def get_qa_meta(helper, split, ds):

    question_char_count = 0
    context_char_count = 0
    answer_count = 0
    answer_char_count = 0
    type_counter = Counter()
    choices_counter = Counter()

    for sample in ds:
        question_char_count += len(sample["question"])
        context_char_count += len(sample["context"])
        type_counter[sample["type"]] += 1
        for choice in sample["choices"]:
            choices_counter[choice] += 1
        for answer in sample["answer"]:
            answer_count += 1
            answer_char_count += len(answer)

    meta = {
        "split": split,
        "samples_count": ds.num_rows,
        "question_char_count": question_char_count,
        "context_char_count": context_char_count,
        "answer_count": answer_count,
        "answer_char_count": answer_char_count,
        "type_counts": dict(type_counter.most_common(MAX_COMMON)),
        "choices_counts": dict(choices_counter.most_common(MAX_COMMON)),
    }

    return meta


def get_t2t_meta(helper, split, ds):

    text_1_char_count = 0
    text_2_char_count = 0
    text_1_name_counter = Counter()
    text_2_name_counter = Counter()

    for sample in ds:
        text_1_char_count += (
            len(sample["text_1"]) if sample["text_1"] is not None else 0
        )
        text_2_char_count += (
            len(sample["text_2"]) if sample["text_2"] is not None else 0
        )
        text_1_name_counter[sample["text_1_name"]] += 1
        text_2_name_counter[sample["text_2_name"]] += 1

    meta = {
        "split": split,
        "samples_count": ds.num_rows,
        "text_1_char_count": text_1_char_count,
        "text_2_char_count": text_2_char_count,
        "text_1_name_counts": dict(text_1_name_counter.most_common(MAX_COMMON)),
        "text_2_name_counts": dict(text_2_name_counter.most_common(MAX_COMMON)),
    }

    return meta


def get_te_meta(helper, split, ds):

    premise_char_count = 0
    hypothesis_char_count = 0
    label_counter = Counter()

    for sample in ds:
        premise_char_count += (
            len(sample["premise"]) if sample["premise"] is not None else 0
        )
        hypothesis_char_count += (
            len(sample["hypothesis"]) if sample["hypothesis"] is not None else 0
        )
        label_counter[sample["label"]] += 1

    meta = {
        "split": split,
        "samples_count": ds.num_rows,
        "premise_char_count": premise_char_count,
        "hypothesis_char_count": hypothesis_char_count,
        "label_counts": dict(label_counter.most_common(MAX_COMMON)),
    }

    return meta


def gather_metadatas_json(conhelps, data_dir_base: Optional[str]=None):

    # gather configs by dataset
    configs_by_ds = defaultdict(list)
    for helper in conhelps:
        configs_by_ds[helper.dataset_name].append(helper)

    # now gather metadata
    dataset_metas = {}
    for dataset_name, helpers in configs_by_ds.items():
        print("dataset_name: ", dataset_name)

        config_metas = {}
        for helper in helpers:
            print("config name: ", helper.config.name)
            if helper.config.name in [
                'bioasq_10b_bigbio_qa',
            ]:
                continue


            if helper.is_local:
                if dataset_name == "psytar":
                    data_dir = os.path.join(data_dir_base, dataset_name, "PsyTAR_dataset.xlsx")
                else:
                    data_dir = os.path.join(data_dir_base, dataset_name)
            else:
                data_dir = None
            dsd = helper.load_dataset(data_dir=data_dir)

            split_metas = {}
            for split, ds in dsd.items():

                if helper.config.schema == "bigbio_kb":
                    meta = get_kb_meta(helper, split, ds)

                elif helper.config.schema == "bigbio_text":
                    meta = get_text_meta(helper, split, ds)

                elif helper.config.schema == "bigbio_t2t":
                    meta = get_t2t_meta(helper, split, ds)

                elif helper.config.schema == "bigbio_pairs":
                    meta = get_pairs_meta(helper, split, ds)

                elif helper.config.schema == "bigbio_qa":
                    meta = get_qa_meta(helper, split, ds)

                elif helper.config.schema == "bigbio_te":
                    meta = get_te_meta(helper, split, ds)

                else:
                    raise ValueError()

                split_metas[split] = meta

            config_meta = {
                "config_name": helper.config.name,
                "bigbio_schema": helper.config.schema,
                "tasks": [el.name for el in helper.tasks],
                "splits": split_metas,
                "splits_count": len(split_metas),
            }
            config_metas[helper.config.name] = config_meta

        dataset_meta = {
            "dataset_name": dataset_name,
            "is_local": False,
            "languages": [el.name for el in helper.languages],
            "bigbio_version": helper.bigbio_version,
            "source_version": helper.source_version,
            "citation": helper.citation,
            "description": json.dumps(helper.description),
            "homepage": helper.homepage,
            "license": helper.license,
            "config_metas": config_metas,
            "configs_count": len(config_metas),
        }
        dataset_metas[dataset_name] = dataset_meta

    return dataset_metas


def flatten_metadatas(dataset_metas):

    # write flat tabular data for each schema
    flat_rows = {
        "kb": [],
        "text": [],
        "t2t": [],
        "pairs": [],
        "qa": [],
        "te": [],
    }

    dataset_row_names = [
        "dataset_name",
        "is_local",
        "languages",
        "bigbio_version",
        "source_version",
        "citation",
        "description",
        "homepage",
        "license",
        "configs_count",
    ]

    config_row_names = [
        "config_name",
        "biogbio_schema",
        "tasks",
        "splits_count",
    ]

    split_row_names = {}

    for dataset_name, dataset_meta in dataset_metas.items():

        dataset_row = [
            dataset_name,
            dataset_meta["is_local"],
            dataset_meta["languages"],
            dataset_meta["bigbio_version"],
            dataset_meta["source_version"],
            dataset_meta["citation"],
            dataset_meta["description"],
            dataset_meta["homepage"],
            dataset_meta["license"],
            dataset_meta["configs_count"],
        ]

        for config_name, config_meta in dataset_meta["config_metas"].items():

            config_row = [
                config_name,
                config_meta["bigbio_schema"],
                config_meta["tasks"],
                config_meta["splits_count"],
            ]

            print(config_name, config_meta["bigbio_schema"])

            for split, split_meta in config_meta["splits"].items():

                if config_meta["bigbio_schema"] == "bigbio_kb":
                    split_row = [
                        split,
                        split_meta["samples_count"],
                        split_meta["passages_count"],
                        json.dumps(split_meta["passages_type_counts"]),
                        split_meta["passages_char_count"],
                        split_meta["entities_count"],
                        json.dumps(split_meta["entities_type_counts"]),
                        json.dumps(split_meta["entities_db_name_counts"]),
                        split_meta["entities_unique_db_id_counts"],
                        split_meta["events_count"],
                        split_meta["events_arguments_count"],
                        json.dumps(split_meta["events_arguments_role_counts"]),
                        split_meta["coreferences_count"],
                        split_meta["relations_count"],
                        json.dumps(split_meta["relations_type_counts"]),
                        json.dumps(split_meta["relations_db_name_counts"]),
                        split_meta["relations_unique_db_id_counts"],
                    ]
                    row = dataset_row + config_row + split_row
                    flat_rows['kb'].append(row)
                    split_row_names['kb'] = list(split_meta.keys())

                elif config_meta["bigbio_schema"] == "bigbio_text":
                    split_row = [
                        split,
                        split_meta["samples_count"],
                        split_meta["text_char_count"],
                        split_meta["labels_count"],
                        json.dumps(split_meta["labels_counts"]),
                    ]
                    row = dataset_row + config_row + split_row
                    flat_rows['text'].append(row)
                    split_row_names['text'] = list(split_meta.keys())

                elif config_meta["bigbio_schema"] == "bigbio_t2t":
                    split_row = [
                        split,
                        split_meta["samples_count"],
                        split_meta["text_1_char_count"],
                        split_meta["text_2_char_count"],
                        json.dumps(split_meta["text_1_name_counts"]),
                        json.dumps(split_meta["text_2_name_counts"]),
                    ]
                    row = dataset_row + config_row + split_row
                    flat_rows['t2t'].append(row)
                    split_row_names['t2t'] = list(split_meta.keys())

                elif config_meta["bigbio_schema"] == "bigbio_pairs":
                    split_row = [
                        split,
                        split_meta["samples_count"],
                        split_meta["text_1_char_count"],
                        split_meta["text_2_char_count"],
                        json.dumps(split_meta["label_counts"]),
                    ]
                    row = dataset_row + config_row + split_row
                    flat_rows['pairs'].append(row)
                    split_row_names['pairs'] = list(split_meta.keys())

                elif config_meta["bigbio_schema"] == "bigbio_qa":
                    split_row = [
                        split,
                        split_meta["samples_count"],
                        split_meta["question_char_count"],
                        split_meta["context_char_count"],
                        split_meta["answer_count"],
                        split_meta["answer_char_count"],
                        json.dumps(split_meta["type_counts"]),
                        json.dumps(split_meta["choices_counts"]),
                    ]
                    row = dataset_row + config_row + split_row
                    flat_rows['qa'].append(row)
                    split_row_names['qa'] = list(split_meta.keys())

                elif config_meta["bigbio_schema"] == "bigbio_te":
                    split_row = [
                        split,
                        split_meta["samples_count"],
                        split_meta["premise_char_count"],
                        split_meta["hypothesis_char_count"],
                        json.dumps(split_meta["label_counts"]),
                    ]
                    row = dataset_row + config_row + split_row
                    flat_rows['te'].append(row)
                    split_row_names['te'] = list(split_meta.keys())

                else:
                    raise ValueError()

    dfs = {
        key: pd.DataFrame(
            flat_rows[key],
            columns = dataset_row_names + config_row_names + split_row_names[key]
        ) for key in split_row_names.keys()
    }
    return dfs



# create a BigBioConfigHelpers
# ==========================================================
conhelps = BigBioConfigHelpers()
conhelps = conhelps.filtered(lambda x: x.dataset_name != "pubtator_central")
conhelps = conhelps.filtered(lambda x: x.is_bigbio_schema)

print(
    "loaded {} configs from {} datasets".format(
        len(conhelps),
        len(set([helper.dataset_name for helper in conhelps])),
    )
)

do_public = True
do_private = True

if do_public:
    public_conhelps = conhelps.filtered(lambda x: not x.is_local)
    public_dataset_metas = gather_metadatas_json(public_conhelps)
    with open("bigbio-public-metadatas.json", "w") as fp:
        json.dump(public_dataset_metas, fp, indent=4)
    public_dfs = flatten_metadatas(public_dataset_metas)
    for key, df in public_dfs.items():
        df.to_parquet(f"bigbio-public-metadatas-flat-{key}.parquet")
        df.to_csv(f"bigbio-public-metadatas-flat-{key}.csv", index=False)

if do_private:
    data_dir_base = "/home/galtay/data/bigbio"
    private_conhelps = conhelps.filtered(lambda x: x.is_local)
    private_dataset_metas = gather_metadatas_json(private_conhelps, data_dir_base=data_dir_base)
    with open("bigbio-private-metadatas.json", "w") as fp:
        json.dump(private_dataset_metas, fp, indent=4)
    private_dfs = flatten_metadatas(private_dataset_metas)
    for key, df in private_dfs.items():
        df.to_parquet(f"bigbio-private-metadatas-flat-{key}.parquet")
        df.to_csv(f"bigbio-private-metadatas-flat-{key}.csv", index=False)
