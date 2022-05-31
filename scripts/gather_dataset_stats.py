"""
Gather dataset statistics
Help make plots
"""
from collections import Counter
from collections import defaultdict, OrderedDict
import dataclasses
import json
import os
from typing import Optional

import pandas as pd

import bigbio
from bigbio.dataloader import BigBioConfigHelpers


MAX_COMMON = 50


def gather_metadatas_json(conhelps, data_dir_base: Optional[str] = None):

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
                "bioasq_10b_bigbio_qa",
            ]:
                continue

            if helper.is_local:
                if dataset_name == "psytar":
                    data_dir = os.path.join(
                        data_dir_base, dataset_name, "PsyTAR_dataset.xlsx"
                    )
                else:
                    data_dir = os.path.join(data_dir_base, dataset_name)
            else:
                data_dir = None

            split_metas_dataclasses = helper.get_metadata(data_dir=data_dir)
            split_metas = {}
            for split, meta_dc in split_metas_dataclasses.items():
                split_metas[split] = dataclasses.asdict(meta_dc)
                split_metas["split"] = split

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
                    flat_rows["kb"].append(row)
                    split_row_names["kb"] = list(split_meta.keys())

                elif config_meta["bigbio_schema"] == "bigbio_text":
                    split_row = [
                        split,
                        split_meta["samples_count"],
                        split_meta["text_char_count"],
                        split_meta["labels_count"],
                        json.dumps(split_meta["labels_counts"]),
                    ]
                    row = dataset_row + config_row + split_row
                    flat_rows["text"].append(row)
                    split_row_names["text"] = list(split_meta.keys())

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
                    flat_rows["t2t"].append(row)
                    split_row_names["t2t"] = list(split_meta.keys())

                elif config_meta["bigbio_schema"] == "bigbio_pairs":
                    split_row = [
                        split,
                        split_meta["samples_count"],
                        split_meta["text_1_char_count"],
                        split_meta["text_2_char_count"],
                        json.dumps(split_meta["label_counts"]),
                    ]
                    row = dataset_row + config_row + split_row
                    flat_rows["pairs"].append(row)
                    split_row_names["pairs"] = list(split_meta.keys())

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
                    flat_rows["qa"].append(row)
                    split_row_names["qa"] = list(split_meta.keys())

                elif config_meta["bigbio_schema"] == "bigbio_te":
                    split_row = [
                        split,
                        split_meta["samples_count"],
                        split_meta["premise_char_count"],
                        split_meta["hypothesis_char_count"],
                        json.dumps(split_meta["label_counts"]),
                    ]
                    row = dataset_row + config_row + split_row
                    flat_rows["te"].append(row)
                    split_row_names["te"] = list(split_meta.keys())

                else:
                    raise ValueError()

    dfs = {
        key: pd.DataFrame(
            flat_rows[key],
            columns=dataset_row_names + config_row_names + split_row_names[key],
        )
        for key in split_row_names.keys()
    }
    return dfs


if __name__ == "__main__":

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
        private_dataset_metas = gather_metadatas_json(
            private_conhelps, data_dir_base=data_dir_base
        )
        with open("bigbio-private-metadatas.json", "w") as fp:
            json.dump(private_dataset_metas, fp, indent=4)
        private_dfs = flatten_metadatas(private_dataset_metas)
        for key, df in private_dfs.items():
            df.to_parquet(f"bigbio-private-metadatas-flat-{key}.parquet")
            df.to_csv(f"bigbio-private-metadatas-flat-{key}.csv", index=False)
