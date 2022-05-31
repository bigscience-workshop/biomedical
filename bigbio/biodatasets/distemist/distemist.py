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

from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks
from bigbio.utils.license import Licenses

_LOCAL = False
_CITATION = """\
@dataset{luis_gasco_2022_6458455,
  author       = {Luis Gasco and Eulàlia Farré and Miranda-Escalada, Antonio and Salvador Lima and Martin Krallinger},
  title        = {{DisTEMIST corpus: detection and normalization of disease mentions in spanish clinical cases}},
  month        = apr,
  year         = 2022,
  note         = {{Funded by the Plan de Impulso de las Tecnologías del Lenguaje (Plan TL).}},
  publisher    = {Zenodo},
  version      = {2.0.0},
  doi          = {10.5281/zenodo.6458455},
  url          = {https://doi.org/10.5281/zenodo.6458455}
}
"""

_DATASETNAME = "distemist"
_DESCRIPTION = """\
The DisTEMIST corpus is a collection of 1000 clinical cases with disease annotations linked with Snomed-CT concepts.
All documents are released in the context of the BioASQ DisTEMIST track for CLEF 2022.
"""

_HOMEPAGE = "https://zenodo.org/record/6458455"
_LICENSE = Licenses.CC_BY_4p0
_URLS = {
    _DATASETNAME: "https://zenodo.org/record/6458455/files/distemist.zip?download=1",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "2.0.0"
_BIGBIO_VERSION = "1.0.0"


class DistemistDataset(datasets.GeneratorBasedBuilder):
    """
    The DisTEMIST corpus is a collection of 1000 clinical cases with disease annotations linked with Snomed-CT
    concepts.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="distemist_source",
            version=SOURCE_VERSION,
            description="DisTEMIST source schema",
            schema="source",
            subset_id="distemist",
        ),
        BigBioConfig(
            name="distemist_bigbio_kb",
            version=BIGBIO_VERSION,
            description="DisTEMIST BigBio schema",
            schema="bigbio_kb",
            subset_id="distemist",
        ),
    ]

    DEFAULT_CONFIG_NAME = "distemist_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "passages": [
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "text": datasets.Sequence(datasets.Value("string")),
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                        }
                    ],
                    "entities": [
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "text": datasets.Sequence(datasets.Value("string")),
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                            "concept_codes": datasets.Sequence(
                                datasets.Value("string")
                            ),
                            "semantic_relations": datasets.Sequence(
                                datasets.Value("string")
                            ),
                        }
                    ],
                }
            )
        elif self.config.schema == "bigbio_kb":
            features = schemas.kb_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "entities_mapping_file_path": Path(data_dir)
                    / "training/subtrack1_entities/distemist_subtrack1_training_mentions.tsv",
                    "linking_mapping_file_path": Path(data_dir)
                    / "training/subtrack2_linking/distemist_subtrack1_training1_linking.tsv",
                    "text_files_dir": Path(data_dir) / "training/text_files",
                },
            ),
        ]

    def _generate_examples(
        self,
        entities_mapping_file_path: Path,
        linking_mapping_file_path: Path,
        text_files_dir: Path,
    ) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        entities_mapping = pd.read_csv(entities_mapping_file_path, sep="\t")
        linking_mapping = pd.read_csv(linking_mapping_file_path, sep="\t")

        entity_file_names = set(entities_mapping["filename"])
        linking_file_names = set(linking_mapping["filename"])

        # entity_file_names = entity_file_names.difference(linking_file_names)

        for uid, filename in enumerate(entity_file_names):
            text_file = text_files_dir / f"{filename}.txt"

            doc_text = text_file.read_text()
            # doc_text = doc_text.replace("\n", "")

            if filename in linking_file_names:
                entities_df: pd.DataFrame = linking_mapping[
                    linking_mapping["filename"] == filename
                ]
            else:
                entities_df: pd.DataFrame = entities_mapping[
                    entities_mapping["filename"] == filename
                ]

            example = {
                "id": f"{uid}",
                "document_id": filename,
                "passages": [
                    {
                        "id": f"{uid}_{filename}_passage",
                        "type": "clinical_case",
                        "text": [doc_text],
                        "offsets": [[0, len(doc_text)]],
                    }
                ],
            }
            if self.config.schema == "bigbio_kb":
                example["events"] = []
                example["coreferences"] = []
                example["relations"] = []

            entities = []
            for row in entities_df.itertuples(name="Entity"):
                entity = {
                    "id": f"{uid}_{row.filename}_{row.Index}_entity_id_{row.mark}",
                    "type": row.label,
                    "text": [row.span],
                    "offsets": [[row.off0, row.off1]],
                }
                if self.config.schema == "source":
                    entity["concept_codes"] = []
                    entity["semantic_relations"] = []
                    if filename in linking_file_names:
                        entity["concept_codes"] = row.code.split("+")
                        entity["semantic_relations"] = row.semantic_rel.split("+")

                elif self.config.schema == "bigbio_kb":
                    entity["normalized"] = []

                entities.append(entity)

            example["entities"] = entities
            yield uid, example
