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
The CellFinder project aims to create a stem cell data repository by linking information from existing public databases
 and by performing text mining on the research literature. The first version of the corpus is composed
 of 10 full text documents containing more than 2,100 sentences, 65,000 tokens and 5,200 annotations for entities.
 The corpus has been annotated with six types of entities (anatomical parts, cell components, cell lines, cell types,
 genes/protein and species) with an overall inter-annotator agreement around 80%.
(see https://www.informatik.hu-berlin.de/de/forschung/gebiete/wbi/resources/cellfinder/).
"""
from pathlib import Path
from typing import Dict, Iterator, Tuple

import datasets

import bigbio.utils.parsing as parsing
from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.license import Licenses
from bigbio.utils.constants import Tasks

_LOCAL = False
_CITATION = """\
@inproceedings{neves2012annotating,
  title={Annotating and evaluating text for stem cell research},
  author={Neves, Mariana and Damaschun, Alexander and Kurtz, Andreas and Leser, Ulf},
  booktitle={Proceedings of the Third Workshop on Building and Evaluation Resources for Biomedical Text Mining\
   (BioTxtM 2012) at Language Resources and Evaluation (LREC). Istanbul, Turkey},
  pages={16--23},
  year={2012},
  organization={Citeseer}
}
"""

_DATASETNAME = "cellfinder"

_DESCRIPTION = """\
The CellFinder project aims to create a stem cell data repository by linking information from existing public databases
 and by performing text mining on the research literature. The first version of the corpus is composed
 of 10 full text documents containing more than 2,100 sentences, 65,000 tokens and 5,200 annotations for entities.
 The corpus has been annotated with six types of entities (anatomical parts, cell components, cell lines, cell types,
 genes/protein and species) with an overall inter-annotator agreement around 80%.
(see https://www.informatik.hu-berlin.de/de/forschung/gebiete/wbi/resources/cellfinder/).
"""

_HOMEPAGE = (
    "https://www.informatik.hu-berlin.de/de/forschung/gebiete/wbi/resources/cellfinder/"
)
_LICENSE_OLD = "CC BY-SA 3.0"

_SOURCE_URL = (
    "https://www.informatik.hu-berlin.de/de/forschung/gebiete/wbi/resources/cellfinder/"
)
_URLS = {
    _DATASETNAME: _SOURCE_URL + "cellfinder1_brat.tar.gz",
    _DATASETNAME + "_splits": _SOURCE_URL + "cellfinder1_brat_sections.tar.gz",
}

_SUPPORTED_TASKS = [
    Tasks.NAMED_ENTITY_RECOGNITION
]  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class CellFinderDataset(datasets.GeneratorBasedBuilder):
    """The CellFinder corpus."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="cellfinder_source",
            version=SOURCE_VERSION,
            description="CellFinder source schema",
            schema="source",
            subset_id="cellfinder",
        ),
        BigBioConfig(
            name="cellfinder_bigbio_kb",
            version=BIGBIO_VERSION,
            description="CellFinder BigBio schema",
            schema="bigbio_kb",
            subset_id="cellfinder",
        ),
        BigBioConfig(
            name="cellfinder_splits_source",
            version=SOURCE_VERSION,
            description="CellFinder source schema",
            schema="source",
            subset_id="cellfinder_splits",
        ),
        BigBioConfig(
            name="cellfinder_splits_bigbio_kb",
            version=BIGBIO_VERSION,
            description="CellFinder BigBio schema",
            schema="bigbio_kb",
            subset_id="cellfinder_splits",
        ),
    ]

    DEFAULT_CONFIG_NAME = "cellfinder_source"
    SPLIT_TO_IDS = {
        "train": [16316465, 17381551, 17389645, 18162134, 18286199],
        "test": [15971941, 16623949, 16672070, 17288595, 17967047],
    }

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "document_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "type": datasets.Value("string"),
                    "entities": [
                        {
                            "entity_id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                            "text": datasets.Sequence(datasets.Value("string")),
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

    def _split_generators(self, dl_manager):
        urls = _URLS[_DATASETNAME]
        if self.config.subset_id.endswith("_splits"):
            urls = _URLS[_DATASETNAME + "_splits"]

        data_dir = Path(dl_manager.download_and_extract(urls))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_dir": data_dir, "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"data_dir": data_dir, "split": "test"},
            ),
        ]

    def _is_to_exclude(self, file: Path) -> bool:

        to_exclude = False

        if (
            file.name.startswith("._")
            or file.name.endswith(".ann")
            or file.name == "LICENSE"
        ):
            to_exclude = True

        return to_exclude

    def _not_in_split(self, file: Path, split: str) -> bool:

        to_exclude = False

        # SKIP files according to split
        if self.config.subset_id.endswith("_splits"):
            file_id = file.stem.split("_")[0]
        else:
            file_id = file.stem

        if int(file_id) not in self.SPLIT_TO_IDS[split]:
            to_exclude = True

        return to_exclude

    def _generate_examples(
        self, data_dir: Path, split: str
    ) -> Iterator[Tuple[str, Dict]]:
        if self.config.schema == "source":
            for file in data_dir.iterdir():

                # Ignore hidden files and annotation files - we just consider the brat text files
                if self._is_to_exclude(file=file):
                    continue

                if self._not_in_split(file=file, split=split):
                    continue

                # Read brat annotations for the given text file and convert example to the source format
                brat_example = parsing.parse_brat_file(file)
                source_example = self._to_source_example(file, brat_example)

                yield source_example["document_id"], source_example

        elif self.config.schema == "bigbio_kb":
            for file in data_dir.iterdir():

                # Ignore hidden files and annotation files - we just consider the brat text files
                if self._is_to_exclude(file=file):
                    continue

                if self._not_in_split(file=file, split=split):
                    continue

                # Read brat annotations for the given text file and convert example to the BigBio-KB format
                brat_example = parsing.parse_brat_file(file)
                kb_example = parsing.brat_parse_to_bigbio_kb(brat_example)
                kb_example["id"] = kb_example["document_id"]

                # Fix text type annotation for the converted example
                kb_example["passages"][0]["type"] = self.get_text_type(file)

                yield kb_example["id"], kb_example

    def _to_source_example(self, input_file: Path, brat_example: Dict) -> Dict:
        """
        Converts an example extracted using the default brat parsing logic to the source format
        of the given corpus.
        """
        text_type = self.get_text_type(input_file)
        source_example = {
            "document_id": brat_example["document_id"],
            "text": brat_example["text"],
            "type": text_type,
        }

        id_prefix = brat_example["document_id"] + "_"

        source_example["entities"] = []
        for entity_annotation in brat_example["text_bound_annotations"]:
            entity_ann = entity_annotation.copy()

            entity_ann["entity_id"] = id_prefix + entity_ann["id"]
            entity_ann.pop("id")

            source_example["entities"].append(entity_ann)

        return source_example

    def get_text_type(self, input_file: Path) -> str:
        """
        Exctracts section name from filename, if absent return full_text
        """

        name_parts = str(input_file.stem).split("_")
        if len(name_parts) == 3:
            return name_parts[2]
        return "full_text"
