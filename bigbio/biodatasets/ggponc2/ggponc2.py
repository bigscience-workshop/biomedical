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

import json
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd
from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks
from bigbio.utils.license import Licenses

_LOCAL = True
_CITATION = """\
@inproceedings{borchert-etal-2022-ggponc,
    title = "{GGPONC} 2.0 - The {G}erman Clinical Guideline Corpus for Oncology: Curation Workflow, Annotation Policy, Baseline {NER} Taggers",
    author = "Borchert, Florian  and
      Lohr, Christina  and
      Modersohn, Luise  and
      Witt, Jonas  and
      Langer, Thomas  and
      Follmann, Markus  and
      Gietzelt, Matthias  and
      Arnrich, Bert  and
      Hahn, Udo  and
      Schapranow, Matthieu-P.",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.389",
    pages = "3650--3660",
}
"""
_DATASETNAME = "ggponc2"
_DESCRIPTION = """\
The GGPONC project aims to provide a freely distributable corpus of German medical text for NLP researchers. 
Clinical guidelines are particularly suitable to create such corpora, as they contain no protected health information 
(PHI), which distinguishes them from other kinds of medical text.

The second version of the corpus (GGPONC 2.0) consists of 30 German oncology guidelines with 1.87 million tokens. 
It has been completely manually annotated on the entity level by 7 medical students using the INCEpTION platform over a 
time frame of 6 months in more than 1200 hours of work. This makes GGPONC 2.0 the largest annotated, freely 
distributable corpus of German medical text at the moment.

Annotated entities are Findings (Diagnosis / Pathology, Other Finding), Substances (Clinical Drug, Nutrients / Body 
Substances, External Substances) and Procedures (Therapeutic, Diagnostic), as well as Specifications for these entities. 
In total, annotators have created more than 200000 entity annotations. In addition, fragment relationships have been 
annotated to explicitly indicate elliptical coordinated noun phrases, a common phenomenon in German text."""
_HOMEPAGE = "https://www.leitlinienprogramm-onkologie.de/projekte/ggponc-english/"
_LANGUAGES = [Lang.DE]
_URLS = {}
_PUBMED = False
_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]
_SOURCE_VERSION = "2.0.0"
_BIGBIO_VERSION = "1.0.0"
_DISPLAYNAME = "GGPONC 2.0"
_DATASETNAME = "ggponc2"
_LICENSE = Licenses.DUA


class GgponcDataset(datasets.GeneratorBasedBuilder):

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)
    DEFAULT_CONFIG_NAME = "ggponc2_fine_long_bigbio_kb"

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="ggponc2_fine_long_bigbio_kb",
            version=BIGBIO_VERSION,
            description="GGPONC 2.0 (fine grained categories and long spans) schema",
            schema="bigbio_kb",
            subset_id="ggponc2",
        ),
        BigBioConfig(
            name="ggponc2_fine_short_bigbio_kb",
            version=BIGBIO_VERSION,
            description="GGPONC 2.0 (fine grained categories and short spans) schema",
            schema="bigbio_kb",
            subset_id="ggponc2",
        ),
        BigBioConfig(
            name="ggponc2_coarse_long_bigbio_kb",
            version=BIGBIO_VERSION,
            description="GGPONC 2.0 (coarse categories and long spans) schema",
            schema="bigbio_kb",
            subset_id="ggponc2",
        ),
        BigBioConfig(
            name="ggponc2_coarse_short_bigbio_kb",
            version=BIGBIO_VERSION,
            description="GGPONC 2.0 (coarse categories and short spans) schema",
            schema="bigbio_kb",
            subset_id="ggponc2",
        ),
    ]

    def _info(self) -> datasets.DatasetInfo:

        features = schemas.kb_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        if self.config.data_dir is None:
            raise ValueError(
                "This is a local dataset. Please pass the data_dir kwarg to load_dataset."
            )
        else:
            data_dir = Path(self.config.data_dir)

        split_dir = data_dir / "annotations/splits.csv"

        dir_lookup = {
            "ggponc2_fine_long_bigbio_kb": data_dir
            / "annotations/json/fine/long/all.json",
            "ggponc2_fine_short_bigbio_kb": data_dir
            / "annotations/json/fine/short/all.json",
            "ggponc2_coarse_long_bigbio_kb": data_dir
            / "annotations/json/coarse/long/all.json",
            "ggponc2_coarse_short_bigbio_kb": data_dir
            / "annotations/json/coarse/short/all.json",
        }

        data_dir = dir_lookup[self.config.name]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": "train",
                    "filepath": data_dir,
                    "split_dir": split_dir,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "split": "test",
                    "filepath": data_dir,
                    "split_dir": split_dir,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "split": "dev",
                    "filepath": data_dir,
                    "split_dir": split_dir,
                },
            ),
        ]

    def _generate_examples(
        self, filepath: str, split: str, split_dir: str
    ) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        splits = pd.read_csv(split_dir)

        with open(filepath, encoding="utf8") as file:
            data = json.load(file)

            for uid, row in enumerate(data):
                file_name = row["document_id"].replace("tsv", "txt")
                file_split = splits.loc[splits["file"] == file_name]["split"].item()
                if file_split == split:
                    out = {
                        "id": uid,
                        "document_id": row["document_id"],
                        "passages": [],
                        "entities": row["entities"],
                    }

                    for j, passage in enumerate(row["passages"]):
                        passage_id = passage["id"]
                        out["passages"].append(
                            {
                                "id": f"{uid}-{passage_id}-{j}",
                                "type": passage["type"],
                                "text": [passage["text"]],
                                "offsets": passage["offsets"],
                            }
                        )

                    for i, _ in enumerate(out["entities"]):
                        out["entities"][i]["id"] = f"{uid}-{i}"
                        out["entities"][i]["normalized"] = []

                    out["events"] = []
                    out["coreferences"] = []
                    out["relations"] = []

                    yield uid, out
