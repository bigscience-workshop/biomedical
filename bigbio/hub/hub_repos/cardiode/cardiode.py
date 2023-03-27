# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from .bigbiohub import BigBioConfig, Tasks, kb_features

_LOCAL = True
_CITATION = """\
@data{data/AFYQDY_2022,
author = {Christoph Dieterich},
publisher = {heiDATA},
title = {{CARDIO:DE}},
year = {2022},
version = {V5},
doi = {10.11588/data/AFYQDY},
url = {https://doi.org/10.11588/data/AFYQDY}
}
"""

_DESCRIPTION = """\
First freely available and distributable large German clinical corpus from the cardiovascular domain.
"""

_HOMEPAGE = "https://heidata.uni-heidelberg.de/dataset.xhtml?persistentId=doi%3A10.11588%2Fdata%2FAFYQDY"

_LICENSE = "DUA"
_LANGUAGES = ["German"]
_URLS = {}
_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]
_SOURCE_VERSION = "5.0.0"
_BIGBIO_VERSION = "1.0.0"
_DATASETNAME = "cardiode"
_DISPLAYNAME = "CARDIO:DE"
_PUBMED = False


class CardioDataset(datasets.GeneratorBasedBuilder):
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="cardiode_source",
            version=SOURCE_VERSION,
            description="CARDIO:DE source schema",
            schema="source",
            subset_id="cardiode",
        ),
        BigBioConfig(
            name="cardiode_bigbio_kb",
            version=BIGBIO_VERSION,
            description="CARDIO:DE BigBio schema",
            schema="bigbio_kb",
            subset_id="cardidoe",
        ),
    ]

    DEFAULT_CONFIG_NAME = "cardiode_bigbio_kb"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "doc_id": datasets.Value("string"),
                    "annotations": [
                        {
                            "text": datasets.Value("string"),
                            "tokens": [
                                {
                                    "id": datasets.Value("string"),
                                    "offsets": datasets.Value("string"),
                                    "text": datasets.Value("string"),
                                    "type": datasets.Value("string"),
                                    "parent_annotation_id": datasets.Value("string"),
                                    "section": datasets.Value("string"),
                                }
                            ],
                        }
                    ],
                }
            )

        elif self.config.schema == "bigbio_kb":
            features = kb_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        if self.config.data_dir is None:
            raise ValueError("This is a local dataset. Please pass the data_dir kwarg to load_dataset.")
        else:
            data_dir = self.config.data_dir

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir),
                    "split": "train",
                },
            )
        ]

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        doc_ids = _sort_files(Path(filepath) / "tsv" / "CARDIODE400_main")
        for uid, doc in enumerate(doc_ids):
            tsv_path = Path(filepath) / "tsv" / "CARDIODE400_main" / f"{doc}"
            df, sentences = _parse_tsv(tsv_path)
            if self.config.schema == "source":
                yield uid, _make_source(uid, doc, df, sentences)
            elif self.config.schema == "bigbio_kb":
                yield uid, _make_bigbio_kb(uid, doc, df, sentences)


def _parse_tsv(path: str) -> pd.DataFrame:
    # read whole .tsv as a string
    with open(path, encoding="utf-8") as file:
        content = file.read()

    # separate doc into sentences
    passages = content.split("#")

    # remove the first line (un-tabbed) of each sentence
    # split sentences into words/tokens
    # and store string sentences for the passages
    sentences = []
    for i, passage in enumerate(passages):
        if passage.split("\n")[0].startswith("Text="):
            sentences.append(passage.split("\n")[0].split("Text=")[1])
        passages[i] = passage.split("\n")[1:]

    # clean empty sentences and tokens
    clean_passages = [[token for token in passage if token != ""] for passage in passages if passage != []]

    # make a dataframe out of the clean tokens
    df = []
    for passage in clean_passages:
        for token in passage:
            df.append(token.split("\t"))

    df = pd.DataFrame(df).rename(
        columns={
            0: "passage_token_id",
            1: "token_offset",
            2: "text",
            3: "label",
            4: "uncertain",
            5: "relation",
            6: "section",
        }
    )

    # correct weird rows were label is NoneType
    df["label"].fillna("_", inplace=True)

    # split passage and token ids
    df[["passage_id", "token_id"]] = df["passage_token_id"].str.split("-", expand=True)

    # split labels and their spans
    # some docs do not have labels spanning various tokens (or they do not have any labels at all)
    if df["label"].apply(lambda x: "[" in x).any():
        df[["lab", "span"]] = df["label"].str.split("[", expand=True)
        df["span"] = df["span"].str.replace("]", "", regex=True)
    else:
        df["lab"] = "_"
        df["span"] = None

    # split start and end offsets and cast to int
    df[["offset_start", "offset_end"]] = df["token_offset"].str.split("-", expand=True)
    df["offset_start"] = df["offset_start"].astype(int)
    df["offset_end"] = df["offset_end"].astype(int)

    # correct offset gaps between tokens
    i = 0
    while i < len(df) - 1:
        gap = df.loc[i + 1]["offset_start"] - df.loc[i]["offset_end"]
        if gap > 1:
            df.loc[i + 1 :, "offset_start"] = df.loc[i + 1 :, "offset_start"] - (gap - 1)
            df.loc[i + 1 :, "offset_end"] = df.loc[i + 1 :, "offset_end"] - (gap - 1)
        i += 1

    return df, sentences


def _make_source(uid: int, doc_id: str, df: pd.DataFrame, sentences: list):
    out = {"doc_id": doc_id, "annotations": []}
    for i, sentence in enumerate(sentences):
        anno = {"text": sentence, "tokens": []}
        chunk = df[df["passage_id"] == str(i + 1)]
        for _, row in chunk.iterrows():
            anno["tokens"].append(
                {
                    "id": row["passage_token_id"],
                    "offsets": row["token_offset"],
                    "text": row["text"],
                    "type": row["label"],
                    "parent_annotation_id": row["relation"],
                    "section": row["section"],
                }
            )
        out["annotations"].append(anno)
    return out


def _make_bigbio_kb(uid: int, doc_id: str, df: pd.DataFrame, sentences: list):
    out = {
        "id": str(uid),
        "document_id": doc_id,
        "passages": [],
        "entities": [],
        "events": [],
        "coreferences": [],
        "relations": [],
    }

    # handle passages
    i, sen_num, offset_mark = 0, 0, 0
    while i < len(df):
        pid = df.iloc[i]["passage_id"]
        passage = df[df["passage_id"] == pid]

        out["passages"].append(
            {
                "id": f"{uid}-{pid}",
                "type": "sentence",
                "text": [sentences[sen_num]],
                "offsets": [[offset_mark, offset_mark + len(sentences[sen_num])]],
            }
        )

        i += len(passage)
        offset_mark += len(sentences[sen_num]) + 1
        sen_num += 1

    # handle entities
    text = " ".join(sentences)
    i = 0
    while i < len(df):
        if df.iloc[i]["lab"] != "_" and df.iloc[i]["span"] is None:
            out["entities"].append(
                {
                    "id": f'{uid}-{df.iloc[i]["passage_token_id"]}',
                    "type": df.iloc[i]["lab"],
                    "text": [text[df.iloc[i]["offset_start"] : df.iloc[i]["offset_end"]]],
                    "offsets": [[df.iloc[i]["offset_start"], df.iloc[i]["offset_end"]]],
                    "normalized": [],
                }
            )
            i += 1
        elif df.iloc[i]["span"] is not None:
            ent = df[df["span"] == df.iloc[i]["span"]]
            out["entities"].append(
                {
                    "id": f'{uid}-{df.iloc[i]["passage_token_id"]}',
                    "type": df.iloc[i]["lab"],
                    "text": [text[ent.iloc[0]["offset_start"] : ent.iloc[-1]["offset_end"]]],
                    "offsets": [[ent.iloc[0]["offset_start"], ent.iloc[-1]["offset_end"]]],
                    "normalized": [],
                }
            )
            i += len(ent)
        else:
            i += 1

    return out


def _sort_files(filepath):
    doc_ids = os.listdir(filepath)
    doc_ids = [int(doc_ids[i].split(".")[0]) for i in range(len(doc_ids))]
    doc_ids = sorted(doc_ids)
    doc_ids = [f"{doc_ids[i]}.tsv" for i in range(len(doc_ids))]
    return doc_ids


# This allows you to run your dataloader with `python [dataset_name].py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    datasets.load_dataset(__file__)
