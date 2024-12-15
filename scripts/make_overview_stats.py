from collections import OrderedDict, Counter
import json
import os

import bigbio
from bigbio.utils.constants import SCHEMA_TO_TASKS, Tasks, TASK_TO_SCHEMA

import numpy as np
import pandas as pd


OUTPUT_DIR = "overview_stats"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCHEMA_LONG = [
    "bigbio_kb",
    "bigbio_text",
    "bigbio_pairs",
    "bigbio_qa",
    "bigbio_te",
    "bigbio_t2t",
]

SCHEMA_UPPER = [el.split("_")[1].upper() for el in SCHEMA_LONG]
SCHEMA_LOWER = [el.split("_")[1].lower() for el in SCHEMA_LONG]


COURSE = [
    "NER", "NED", "EE", "COREF", "RE",
    "TXTCLASS",
    "STS",
    "QA",
    "TE",
    "TRANSL", "SUM", "PARA",
]

SCHEMA_UPPER_TO_COURSE = {
    "KB": ["NER", "NED", "EE", "COREF", "RE"],
    "TEXT": ["TXTCLASS"],
    "PAIRS": ["STS"],
    "T2T": ["TRANSL", "SUM", "PARA"],
    "TE": ["TE"],
    "QA": ["QA"],
}

COURSE_TO_SCHEMA_UPPER = {}
for k, vv in SCHEMA_UPPER_TO_COURSE.items():
    for v in vv:
        COURSE_TO_SCHEMA_UPPER[v] = k

COURSE_LONG_TO_SHORT = {
    task.name: task.value for task in Tasks}




COMMON_META_COLUMNS = [
    "dataset_name",
    "is_pubmed",
    "is_local",
    "languages",
    "bigbio_version",
    "source_version",
    "citation",
    "description",
    "homepage",
    "license",
    "configs_count",
    "config_name",
    "bigbio_schema",
    "tasks",
    "splits_count",
    "samples_count",
    "split",
]





def choose_one_fold(df):
    """
    several datasets have multiple crossval folds
    some are split at the config_name level
    others are at the split level
    """

    #-----------------------------------------------------------
    bmask1 = df["dataset_name"].isin(["gad", "pubmed_qa"])
    bmask2 = df["config_name"].str.contains("fold0")
    bmask3 = ~bmask1
    df = df[(bmask1 & bmask2) | bmask3]

    #-----------------------------------------------------------
    bmask1 = df["dataset_name"].isin(["ask_a_patient", "twadrl", "progene"])
    bmask2 = df["split"].str.contains("0")
    bmask3 = ~bmask1
    df = df[(bmask1 & bmask2) | bmask3]

    return df


def build_flat_meta_dfs(one_kfold=True):
    """Read the flat schema specific metadata"""
    flat_meta_dfs = {}
    for short_name in SCHEMA_LOWER:
        pub_file_name = f"metadatas/bigbio-public-metadatas-flat-{short_name}.parquet"
        df_pub = pd.read_parquet(pub_file_name)

        prv_file_name = f"metadatas/bigbio-private-metadatas-flat-{short_name}.parquet"
        if os.path.exists(prv_file_name):
            df_prv = pd.read_parquet(prv_file_name)
        else:
            df_prv = pd.DataFrame(columns=df_pub.columns)

        df = pd.concat([df_pub, df_prv])
        if one_kfold:
            df = choose_one_fold(df)
        flat_meta_dfs[short_name] = df

    return flat_meta_dfs


def build_common_meta_df(flat_meta_dfs):
    df = pd.concat(
        [
            flat_meta_df[COMMON_META_COLUMNS]
            for flat_meta_df in flat_meta_dfs.values()
        ]
    ).reset_index(drop=True)
    return df



# read baseline dataframes
#=====================================================
flat_meta_dfs = build_flat_meta_dfs()
common_df = build_common_meta_df(flat_meta_dfs)


# one row per dataset
#=====================================================
print("=" * 60)
print(" one row per dataset")
print("=" * 60)
ds1 = common_df.drop_duplicates(subset=["dataset_name"])
num_unique_ds = ds1.shape[0]
print("num unique datasets: ", num_unique_ds)
print()

# one row per (dataset, schema)
#=====================================================
print("=" * 60)
print(" one row per (dataset, schema)")
print("=" * 60)
sch1 = common_df.drop_duplicates(subset=["dataset_name", "bigbio_schema"])
num_unique_ds_schema = sch1.shape[0]
print("num unique (dataset, schema): ", num_unique_ds_schema)
print()
print("schema counts: \n", sch1['bigbio_schema'].value_counts())
print()

# one row per (dataset, task)
#=====================================================
print("=" * 60)
print(" one row per (dataset, task)")
print("=" * 60)
tsk1 = common_df.explode("tasks")
tsk1 = tsk1[~tsk1["tasks"].isna()]
tsk1 = tsk1.drop_duplicates(subset=["dataset_name", "tasks"])
num_unique_ds_tasks = tsk1.shape[0]
print("num unique (datasets, tasks): ", num_unique_ds_tasks)
print()
print("task counts: \n", tsk1['bigbio_schema'].value_counts())
print()
print("task counts: \n", tsk1['tasks'].value_counts())
print()

# one row per (dataset, language)
#=====================================================
lng1 = df_report.explode("languages")
lng1 = lng1[~lng1["languages"].isna()]
lng1 = lng1.drop_duplicates(subset=["dataset_name", "languages"])
num_unique_ds_languages = lng1.shape[0]
print("num unique (datasets, languages): ", num_unique_ds_languages)
print("lang counts: \n", lng1['bigbio_schema'].value_counts())
print()
print("lang counts: \n", lng1['languages'].value_counts())
print()
