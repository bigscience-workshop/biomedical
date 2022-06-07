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

"""
The "Psychiatric Treatment Adverse Reactions" (PsyTAR) dataset contains 891 drugs
reviews posted by patients on "askapatient.com", about the effectiveness and adverse
drug events associated with Zoloft, Lexapro, Cymbalta, and Effexor XR.

For each drug review, patient demographics, duration of treatment, and satisfaction
with the drugs were reported.

This dataset can be used for:

1. (multi-label) sentence classification, across 5 labels:
    Adverse Drug Reaction (ADR)
    Withdrawal Symptoms (WDs)
    Sign/Symptoms/Illness (SSIs)
    Drug Indications (DIs)
    Drug Effectiveness (EF)
    Drug Infectiveness (INF)
    and Others (not applicable)

2. Recognition of 5 different types of entity:
    ADRs (4813 mentions)
    WDs (590 mentions)
    SSIs (1219 mentions)
    DIs (792 mentions)

In the source schema, systematic annotation with UMLS and SNOMED-CT concepts are provided.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from bigbio.utils import parsing, schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tags, Tasks
from bigbio.utils.license import Licenses

_TAGS = []
_LANGUAGES = [Lang.EN]
_PUBMED = False
_LOCAL = True
_CITATION = """\
@article{Zolnoori2019,
  author    = {Maryam Zolnoori and
               Kin Wah Fung and
               Timothy B. Patrick and
               Paul Fontelo and
               Hadi Kharrazi and
               Anthony Faiola and
               Yi Shuan Shirley Wu and
               Christina E. Eldredge and
               Jake Luo and
               Mike Conway and
               Jiaxi Zhu and
               Soo Kyung Park and
               Kelly Xu and
               Hamideh Moayyed and
               Somaieh Goudarzvand},
  title     = {A systematic approach for developing a corpus of patient \
               reported adverse drug events: A case study for {SSRI} and {SNRI} medications},
  journal   = {Journal of Biomedical Informatics},
  volume    = {90},
  year      = {2019},
  url       = {https://doi.org/10.1016/j.jbi.2018.12.005},
  doi       = {10.1016/j.jbi.2018.12.005},
}
"""

_DATASETNAME = "psytar"

_DESCRIPTION = """\
The "Psychiatric Treatment Adverse Reactions" (PsyTAR) dataset contains 891 drugs
reviews posted by patients on "askapatient.com", about the effectiveness and adverse
drug events associated with Zoloft, Lexapro, Cymbalta, and Effexor XR.

This dataset can be used for (multi-label) sentence classification of Adverse Drug
Reaction (ADR), Withdrawal Symptoms (WDs), Sign/Symptoms/Illness (SSIs), Drug
Indications (DIs), Drug Effectiveness (EF), Drug Infectiveness (INF) and Others, as well
as for recognition of 5 different types of named entity (in the categories ADRs, WDs,
SSIs and DIs)
"""

_HOMEPAGE = "https://www.askapatient.com/research/pharmacovigilance/corpus-ades-psychiatric-medications.asp"

_LICENSE = Licenses.CC_BY_4p0

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.TEXT_CLASSIFICATION]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


@dataclass
class PsyTARBigBioConfig(BigBioConfig):
    schema: str = "source"
    name: str = "psytar_source"
    version: datasets.Version = _SOURCE_VERSION
    description: str = "PsyTAR source schema"
    subset_id: str = "psytar"


class PsyTARDataset(datasets.GeneratorBasedBuilder):
    """The PsyTAR dataset contains patient's reviews on the effectiveness and adverse
    drug events associated with Zoloft, Lexapro, Cymbalta, and Effexor XR."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        PsyTARBigBioConfig(
            name="psytar_source",
            version=SOURCE_VERSION,
            description="PsyTAR source schema",
            schema="source",
            subset_id="psytar",
        ),
        PsyTARBigBioConfig(
            name="psytar_bigbio_kb",
            version=BIGBIO_VERSION,
            description="PsyTAR BigBio KB schema",
            schema="bigbio_kb",
            subset_id="psytar",
        ),
        PsyTARBigBioConfig(
            name="psytar_bigbio_text",
            version=BIGBIO_VERSION,
            description="PsyTAR BigBio text classification schema",
            schema="bigbio_text",
            subset_id="psytar",
        ),
    ]

    BUILDER_CONFIG_CLASS = PsyTARBigBioConfig

    DEFAULT_CONFIG_NAME = "psytar_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "doc_id": datasets.Value("string"),
                    "disorder": datasets.Value("string"),
                    "side_effect": datasets.Value("string"),
                    "comment": datasets.Value("string"),
                    "gender": datasets.Value("string"),
                    "age": datasets.Value("int32"),
                    "dosage_duration": datasets.Value("string"),
                    "date": datasets.Value("string"),
                    "category": datasets.Value("string"),
                    "sentences": [
                        {
                            "text": datasets.Value("string"),
                            "label": datasets.Sequence([datasets.Value("string")]),
                            "findings": datasets.Value("string"),
                            "others": datasets.Value("string"),
                            "rating": datasets.Value("string"),
                            "category": datasets.Value("string"),
                            "entities": [
                                {
                                    "text": datasets.Value("string"),
                                    "type": datasets.Value("string"),
                                    "mild": datasets.Value("string"),
                                    "moderate": datasets.Value("string"),
                                    "severe": datasets.Value("string"),
                                    "persistent": datasets.Value("string"),
                                    "non_persistent": datasets.Value("string"),
                                    "body_site": datasets.Value("string"),
                                    "rating": datasets.Value("string"),
                                    "drug": datasets.Value("string"),
                                    "class": datasets.Value("string"),
                                    "entity_type": datasets.Value("string"),
                                    "UMLS": datasets.Sequence(
                                        [datasets.Value("string")]
                                    ),
                                    "SNOMED": datasets.Sequence(
                                        [datasets.Value("string")]
                                    ),
                                }
                            ],
                        }
                    ],
                }
            )
        elif self.config.schema == "bigbio_kb":
            features = schemas.kb_features
        elif self.config.schema == "bigbio_text":
            features = schemas.text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        if self.config.data_dir is None:
            raise ValueError(
                "This is a local dataset. Please pass the data_dir kwarg to load_dataset."
            )
        else:
            data_dir = self.config.data_dir

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": Path(data_dir),
                },
            ),
        ]

    def _extract_labels(self, row):
        label = [
            "ADR" * row.ADR,
            "WD" * row.WD,
            "EF" * row.EF,
            "INF" * row.INF,
            "SSI" * row.SSI,
            "DI" * row.DI,
            "Others" * row.others,
        ]
        label = [_l for _l in label if _l != ""]
        return label

    def _columns_to_list(self, row, sheet="ADR"):
        annotations = []
        for i in range(30 if sheet == "ADR" else 10):
            annotations.append(row[f"{sheet}{i + 1}"])
        annotations = [a for a in annotations if not pd.isna(a)]
        return annotations

    def _columns_to_bigbio_kb(self, row, sheet="ADR"):
        annotations = []
        for i in range(30 if sheet == "ADR" else 10):
            annotation = row[f"{sheet}{i + 1}"]
            if not pd.isna(annotation):
                start_index = row.sentences.lower().find(annotation.lower())
                if start_index != -1:
                    end_index = start_index + len(annotation)
                    entity = {
                        "id": f"T{i+1}",
                        "offsets": [[start_index, end_index]],
                        "text": [annotation],
                        "type": sheet,
                    }

                    annotations.append(entity)
        return annotations

    def _standards_columns_to_list(self, row, standard="UMLS"):
        standards = {"UMLS": ["UMLS1", "UMLS2"], "SNOMED": ["SNOMED-CT", "SNOMED-CT.1"]}
        _out_list = []
        for s in standards[standard]:
            _out_list.append(row[s])
        _out_list = [a for a in _out_list if not pd.isna(a)]
        return _out_list

    def _read_sentence_xlsx(self, filepath: Path) -> pd.DataFrame:
        sentence_df = pd.read_excel(
            filepath,
            sheet_name="Sentence_Labeling",
            dtype={"drug_id": str, "sentences": str},
        )

        sentence_df = sentence_df.dropna(subset=["sentences"])
        sentence_df = sentence_df.loc[
            sentence_df.sentences.apply(lambda x: len(x.strip())) > 0
        ]
        sentence_df = sentence_df.fillna(0)

        sentence_df[["ADR", "WD", "EF", "INF", "SSI", "DI"]] = (
            sentence_df[["ADR", "WD", "EF", "INF", "SSI", "DI"]]
            .replace(re.compile("[!* ]+"), 1)
            .astype(int)
        )

        sentence_df["sentence_index"] = sentence_df["sentence_index"].astype("int32")
        sentence_df["drug_id"] = sentence_df["drug_id"].astype("str")

        return sentence_df

    def _read_samples_xlsx(self, filepath: Path) -> pd.DataFrame:
        samples_df = pd.read_excel(
            filepath, sheet_name="Sample", dtype={"drug_id": str}
        )
        samples_df["age"] = samples_df["age"].fillna(0).astype(int)
        samples_df["drug_id"] = samples_df["drug_id"].astype("str")

        return samples_df

    def _read_identified_xlsx_to_bigbio_kb(self, filepath: Path) -> Dict:
        sheet_names = ["ADR", "WD", "SSI", "DI"]
        identified_entities = {}

        for sheet in sheet_names:
            identified_entities[sheet] = pd.read_excel(
                filepath, sheet_name=sheet + "_Identified"
            )
            identified_entities[sheet]["bigbio_kb"] = identified_entities[sheet].apply(
                lambda x: self._columns_to_bigbio_kb(x, sheet), axis=1
            )

        return identified_entities

    TYPE_TO_COLNAME = {"ADR": "ADRs", "DI": "DIs", "SSI": "SSI", "WD": "WDs"}

    def _identified_mapped_xlsx_to_df(self, filepath: Path) -> pd.DataFrame:
        sheet_names_mapped = [
            ["ADR_Mapped", "ADR"],
            ["WD-Mapped ", "WD"],
            ["SSI_Mapped", "SSI"],
            ["DI_Mapped", "DI"],
        ]

        _mappings = []

        # Read the specific XLSX sheet with _Mapped annotations
        for sheet, sheet_short in sheet_names_mapped:
            _df_mapping = pd.read_excel(filepath, sheet_name=sheet)

            # Correcting column names
            if sheet_short in ["WD"]:
                _df_mapping = _df_mapping.rename(
                    columns={"sentence_id": "sentence_index"}
                )

            # Changing column names to allow concatenation
            _df_mapping = _df_mapping.rename(
                columns={self.TYPE_TO_COLNAME[sheet_short]: "entity"}
            )

            # Putting UMLS and SNOMED annotations in a single column
            _df_mapping["UMLS"] = _df_mapping.apply(
                lambda x: self._standards_columns_to_list(x), axis=1
            )
            _df_mapping["SNOMED"] = _df_mapping.apply(
                lambda x: self._standards_columns_to_list(x, standard="SNOMED"), axis=1
            )

            _mappings.append(_df_mapping)

        df_mappings = pd.concat(_mappings).fillna(0)
        df_mappings["sentence_index"] = df_mappings["sentence_index"].astype("int32")
        df_mappings["drug_id"] = df_mappings["drug_id"].astype("str")

        return df_mappings

    def _convert_xlsx_to_source(self, filepath: Path) -> Dict:
        # Read XLSX files
        df_sentences = self._read_sentence_xlsx(filepath)
        df_sentences["label"] = df_sentences.apply(
            lambda x: self._extract_labels(x), axis=1
        )
        df_mappings = self._identified_mapped_xlsx_to_df(filepath)
        df_samples = self._read_samples_xlsx(filepath)

        # Configure indices
        df_samples = df_samples.set_index("drug_id").sort_index()
        df_sentences = df_sentences.set_index(
            ["drug_id", "sentence_index"]
        ).sort_index()
        df_mappings = df_mappings.set_index(["drug_id", "sentence_index"]).sort_index()

        # Iterate over samples
        for sample_row_id, sample in df_samples.iterrows():
            sentences = []
            try:
                df_sentence_selection = df_sentences.loc[sample_row_id]

                # Iterate over sentences
                for sentence_row_id, sentence in df_sentence_selection.iterrows():
                    entities = []
                    try:
                        df_mapped_selection = df_mappings.loc[
                            sample_row_id, sentence_row_id
                        ]

                        # Iterate over entities per sentence
                        for mapped_row_id, row in df_mapped_selection.iterrows():
                            entities.append(
                                {
                                    "text": row["entity"],
                                    "UMLS": row.UMLS,
                                    "SNOMED": row.SNOMED,
                                    "entity_type": row.entity_type,
                                    "type": row.type,
                                    "class": row["class"],
                                    "drug": row.drug,
                                    "rating": row.rating,
                                    "body_site": row["body-site"],
                                    "non_persistent": row["not-persistent"],
                                    "persistent": row["persistent"],
                                    "severe": row.severe,
                                    "moderate": row.moderate,
                                    "mild": row.mild,
                                }
                            )
                    except KeyError:
                        pass

                    sentences.append(
                        {
                            "text": sentence.sentences,
                            "entities": entities,
                            "label": sentence.label,
                            "findings": sentence.Findings,
                            "others": sentence.others,
                            "rating": sentence.rating,
                            "category": sentence.category,
                        }
                    )
            except KeyError:
                pass

            example = {
                "id": sample_row_id,
                "doc_id": sample_row_id,
                "disorder": sample.disorder,
                "side_effect": sample["side-effect"],
                "comment": sample.comment,
                "gender": sample.gender,
                "age": sample.age,
                "dosage_duration": sample.dosage_duration,
                "date": str(sample.date),
                "category": sample.category,
                "sentences": sentences,
            }
            yield example

    def _convert_xlsx_to_bigbio_kb(self, filepath: Path) -> Dict:
        bigbio_kb = self._read_identified_xlsx_to_bigbio_kb(filepath)

        i_doc = 0
        for _, df in bigbio_kb.items():
            for _, row in df.iterrows():
                text = row.sentences
                entities = row["bigbio_kb"]
                doc_id = f"{row['drug_id']}_{row['sentence_index']}_{i_doc}"

                if len(entities) != 0:
                    example = parsing.brat_parse_to_bigbio_kb(
                        {
                            "document_id": doc_id,
                            "text": text,
                            "text_bound_annotations": entities,
                            "normalizations": [],
                            "events": [],
                            "relations": [],
                            "equivalences": [],
                            "attributes": [],
                        },
                    )
                    example["id"] = i_doc
                    i_doc += 1
                    yield example

    def _convert_xlsx_to_bigbio_text(self, filepath: Path) -> Dict:
        df = self._read_sentence_xlsx(filepath)
        df["label"] = df.apply(lambda x: self._extract_labels(x), axis=1)

        for idx, row in df.iterrows():
            example = {
                "id": idx,
                "document_id": f"{row['drug_id']}_{row['sentence_index']}",
                "text": row["label"],
                "labels": row["category"],
            }
            yield example

    def _generate_examples(self, filepath) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        if self.config.schema == "source":
            examples = self._convert_xlsx_to_source(filepath)

        elif self.config.schema == "bigbio_kb":
            examples = self._convert_xlsx_to_bigbio_kb(filepath)

        elif self.config.schema == "bigbio_text":
            examples = self._convert_xlsx_to_bigbio_text(filepath)

        for idx, example in enumerate(examples):
            yield idx, example
