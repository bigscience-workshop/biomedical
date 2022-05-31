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
A corpus for plant and chemical entities and for the relationships between them. The corpus contains 2218 plant
and chemical entities and 600 plant-chemical relationships which are drawn from 1109 sentences in 245 PubMed
abstracts.
"""
from pathlib import Path
from typing import Dict, Iterator, Tuple

import datasets

import bigbio.utils.parsing as parsing
import bigbio.utils.schemas as schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.license import Licenses
from bigbio.utils.constants import Tasks

_LOCAL = False
_CITATION = """\
@article{choi2016corpus,
  title={A corpus for plant-chemical relationships in the biomedical domain},
  author={Choi, Wonjun and Kim, Baeksoo and Cho, Hyejin and Lee, Doheon and Lee, Hyunju},
  journal={BMC bioinformatics},
  volume={17},
  number={1},
  pages={1--15},
  year={2016},
  publisher={Springer}
}
"""

_DATASETNAME = "pcr"

_DESCRIPTION = """
A corpus for plant / herb and chemical entities and for the relationships between them. The corpus contains 2218 plant
and chemical entities and 600 plant-chemical relationships which are drawn from 1109 sentences in 245 PubMed abstracts.
"""

_HOMEPAGE = "http://210.107.182.73/plantchemcorpus.htm"
_LICENSE = Licenses.
_LICENSE_OLD = ""

_URLS = {_DATASETNAME: "http://210.107.182.73/1109_corpus_units_STformat.tar"}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.EVENT_EXTRACTION]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class PCRDataset(datasets.GeneratorBasedBuilder):
    """
    The corpus of plant-chemical relation consists of plants / herbs and chemicals and relations
    between them.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="pcr_source",
            version=SOURCE_VERSION,
            description="PCR source schema",
            schema="source",
            subset_id="pcr",
        ),
        BigBioConfig(
            name="pcr_fixed_source",
            version=SOURCE_VERSION,
            description="PCR (with fixed offsets) source schema",
            schema="source",
            subset_id="pcr_fixed",
        ),
        BigBioConfig(
            name="pcr_bigbio_kb",
            version=BIGBIO_VERSION,
            description="PCR BigBio schema",
            schema="bigbio_kb",
            subset_id="pcr",
        ),
    ]

    DEFAULT_CONFIG_NAME = "pcr_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "document_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "entities": [
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                            "text": datasets.Sequence(datasets.Value("string")),
                            "normalized": [
                                {
                                    "db_name": datasets.Value("string"),
                                    "db_id": datasets.Value("string"),
                                }
                            ],
                        }
                    ],
                    "events": [
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            # refers to the text_bound_annotation of the trigger
                            "trigger": {
                                "text": datasets.Sequence(datasets.Value("string")),
                                "offsets": datasets.Sequence([datasets.Value("int32")]),
                            },
                            "arguments": [
                                {
                                    "role": datasets.Value("string"),
                                    "ref_id": datasets.Value("string"),
                                }
                            ],
                        }
                    ],
                },
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
        data_dir = Path(dl_manager.download_and_extract(urls))
        data_dir = data_dir / "1109 corpus units"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_dir": data_dir},
            )
        ]

    def _generate_examples(self, data_dir: Path) -> Iterator[Tuple[str, Dict]]:
        if self.config.schema == "source":
            for file in data_dir.iterdir():
                if not str(file).endswith(".txt"):
                    continue

                example = parsing.parse_brat_file(file)
                example = parsing.brat_parse_to_bigbio_kb(example)
                example = self._to_source_example(example)

                # Three documents have incorrect offsets - fix them for fixed_source scheme
                if self.config.subset_id == "pcr_fixed" and example["document_id"] in [
                    "463",
                    "509",
                    "566",
                ]:
                    example = self._fix_example(example)

                yield example["document_id"], example

        elif self.config.schema == "bigbio_kb":
            for file in data_dir.iterdir():
                if not str(file).endswith(".txt"):
                    continue

                example = parsing.parse_brat_file(file)
                example = parsing.brat_parse_to_bigbio_kb(example)

                document_id = example["document_id"]
                example["id"] = document_id

                # Three documents have incorrect offsets - fix them for BigBio scheme
                if document_id in ["463", "509", "566"]:
                    example = self._fix_example(example)

                yield example["id"], example

    def _to_source_example(self, bigbio_example: Dict) -> Dict:
        """
        Converts an example in BigBio-KB scheme to an example according to the source scheme
        """
        source_example = bigbio_example.copy()
        source_example["text"] = bigbio_example["passages"][0]["text"][0]

        source_example.pop("passages", None)
        source_example.pop("relations", None)
        source_example.pop("coreferences", None)

        return source_example

    def _fix_example(self, example: Dict) -> Dict:
        """
        Fixes by the example by adapting the offsets of the trigger word of the first
        event. In the official annotation data the end offset is incorrect (for 3 examples).
        """
        first_event = example["events"][0]
        trigger_text = first_event["trigger"]["text"][0]
        offsets = first_event["trigger"]["offsets"][0]

        real_offsets = [offsets[0], offsets[0] + len(trigger_text)]
        example["events"][0]["trigger"]["offsets"] = [real_offsets]

        return example
