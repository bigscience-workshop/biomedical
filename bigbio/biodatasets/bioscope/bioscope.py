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
BioScope
---
The corpus consists of three parts, namely medical free texts, biological full papers and biological scientific
abstracts. The dataset contains annotations at the token level for negative and speculative keywords and at the
sentence level for their linguistic scope. The annotation process was carried out by two independent linguist
annotators and a chief linguist – also responsible for setting up the annotation guidelines – who resolved cases
where the annotators disagreed. The resulting corpus consists of more than 20.000 sentences that were considered
for annotation and over 10% of them actually contain one (or more) linguistic annotation suggesting negation or
uncertainty.
"""

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks

_LANGUAGES = [Lang.EN]
_LOCAL = False
_CITATION = """\
@article{vincze2008bioscope,
  title={The BioScope corpus: biomedical texts annotated for uncertainty, negation and their scopes},
  author={Vincze, Veronika and Szarvas, Gy{\"o}rgy and Farkas, Rich{\'a}rd and M{\'o}ra, Gy{\"o}rgy and Csirik, J{\'a}nos},
  journal={BMC bioinformatics},
  volume={9},
  number={11},
  pages={1--9},
  year={2008},
  publisher={BioMed Central}
}
"""

_DATASETNAME = "bioscope"

_DESCRIPTION = """\
The BioScope corpus consists of medical and biological texts annotated for negation, speculation and their linguistic 
scope. This was done to allow a comparison between the development of systems for negation/hedge detection and scope 
resolution. The BioScope corpus was annotated by two independent linguists following the guidelines written by our 
linguist expert before the annotation of the corpus was initiated.
"""

_HOMEPAGE = "https://rgai.inf.u-szeged.hu/node/105"

_LICENSE = "Creative Commons Attribution 2.0 International (CC BY 2.0)"

_URLS = {
    _DATASETNAME: "https://rgai.sed.hu/sites/rgai.sed.hu/files/bioscope.zip",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class BioscopeDataset(datasets.GeneratorBasedBuilder):
    """The BioScope corpus consists of medical and biological texts annotated for negation, speculation and their linguistic scope."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="bioscope_source",
            version=SOURCE_VERSION,
            description="bioscope source schema",
            schema="source",
            subset_id="bioscope",
        ),
        BigBioConfig(
            name="bioscope_abstracts_source",
            version=SOURCE_VERSION,
            description="bioscope source schema",
            schema="source",
            subset_id="bioscope_abstracts",
        ),
        BigBioConfig(
            name="bioscope_papers_source",
            version=SOURCE_VERSION,
            description="bioscope source schema",
            schema="source",
            subset_id="bioscope_papers",
        ),
        BigBioConfig(
            name="bioscope_medical_texts_source",
            version=SOURCE_VERSION,
            description="bioscope source schema",
            schema="source",
            subset_id="bioscope_medical_texts",
        ),
        BigBioConfig(
            name="bioscope_bigbio_kb",
            version=BIGBIO_VERSION,
            description="bioscope BigBio schema",
            schema="bigbio_kb",
            subset_id="bioscope",
        ),
        BigBioConfig(
            name="bioscope_abstracts_bigbio_kb",
            version=BIGBIO_VERSION,
            description="bioscope BigBio schema",
            schema="bigbio_kb",
            subset_id="bioscope_abstracts",
        ),
        BigBioConfig(
            name="bioscope_papers_bigbio_kb",
            version=BIGBIO_VERSION,
            description="bioscope BigBio schema",
            schema="bigbio_kb",
            subset_id="bioscope_papers",
        ),
        BigBioConfig(
            name="bioscope_medical_texts_bigbio_kb",
            version=BIGBIO_VERSION,
            description="bioscope BigBio schema",
            schema="bigbio_kb",
            subset_id="bioscope_medical_texts",
        ),
    ]

    DEFAULT_CONFIG_NAME = "bioscope_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "document_id": datasets.Value("string"),
                    "document_type": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "entities": [
                        {
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                            "text": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "id": datasets.Value("string"),
                            "normalized": [
                                {
                                    "db_name": datasets.Value("string"),
                                    "db_id": datasets.Value("string"),
                                }
                            ],
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
            license=_LICENSE,
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
                    "data_files": data_dir,
                },
            )
        ]

    def _generate_examples(self, data_files: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        sentences = self._load_sentences(data_files)
        if self.config.schema == "source":
            for guid, sentence_tuple in enumerate(sentences):
                document_type, sentence = sentence_tuple
                example = self._create_example(sentence_tuple)
                example["document_type"] = f"{document_type}_{sentence.attrib['id']}"
                example["text"] = "".join(sentence_tuple[1].itertext())
                yield guid, example

        elif self.config.schema == "bigbio_kb":
            for guid, sentence_tuple in enumerate(sentences):
                document_type, sentence = sentence_tuple
                example = self._create_example(sentence_tuple)
                example["id"] = guid
                example["passages"] = [
                    {
                        "id": f"{document_type}_{sentence.attrib['id']}",
                        "type": document_type,
                        "text": ["".join(sentence.itertext())],
                        "offsets": [(0, len("".join(sentence.itertext())))],
                    }
                ]
                example["events"] = []
                example["coreferences"] = []
                example["relations"] = []
                yield guid, example

    def _load_sentences(self, data_files: Path) -> List:
        """
        Returns a list of tuples (Document type, iterator from dataset)
        """
        if self.config.subset_id.__contains__("abstracts"):
            sentences = self._concat_iterators(
                ("Abstract", ET.parse(os.path.join(data_files, "abstracts.xml")).getroot().iter("sentence"))
            )
        elif self.config.subset_id.__contains__("papers"):
            sentences = self._concat_iterators(
                ("Paper", ET.parse(os.path.join(data_files, "full_papers.xml")).getroot().iter("sentence"))
            )
        elif self.config.subset_id.__contains__("medical_texts"):
            sentences = self._concat_iterators(
                (
                    "Medical text",
                    ET.parse(os.path.join(data_files, "clinical_merger/clinical_records_anon.xml"))
                    .getroot()
                    .iter("sentence"),
                )
            )
        else:
            abstracts = ET.parse(os.path.join(data_files, "abstracts.xml")).getroot().iter("sentence")
            papers = ET.parse(os.path.join(data_files, "full_papers.xml")).getroot().iter("sentence")
            medical_texts = (
                ET.parse(os.path.join(data_files, "clinical_merger/clinical_records_anon.xml"))
                .getroot()
                .iter("sentence")
            )
            sentences = self._concat_iterators(
                ("Abstract", abstracts), ("Paper", papers), ("Medical text", medical_texts)
            )
        return sentences

    @staticmethod
    def _concat_iterators(*iterator_tuple):
        for document_type, iterator in iterator_tuple:
            for element in iterator:
                yield document_type, element

    def _create_example(self, sentence_tuple):
        document_type, sentence = sentence_tuple
        document_type_prefix = document_type[0]

        example = {}
        example["document_id"] = f"{document_type_prefix}_{sentence.attrib['id']}"
        example["entities"] = self._extract_entities(sentence, document_type_prefix)
        return example

    def _extract_entities(self, sentence, document_type_prefix):
        text = "".join(sentence.itertext())
        entities = []
        xcopes = dict([(xcope.attrib["id"], xcope) for xcope in sentence.iter("xcope")])
        cues = dict([(cue.attrib["ref"], cue) for cue in sentence.iter("cue")])
        for idx, xcope in xcopes.items():
            # X2.140.2 has no annotation in raw data
            if cues.get(idx) is None:
                continue
            entities.append(
                {
                    "id": f"{document_type_prefix}_{idx}",
                    "type": cues.get(idx).attrib["type"],
                    "text": ["".join(xcope.itertext())],
                    "offsets": self._extract_offsets(text=text, entity_text="".join(xcope.itertext())),
                    "normalized": [],
                }
            )
        return entities

    def _extract_offsets(self, text, entity_text):
        return [(text.find(entity_text), text.find(entity_text) + len(entity_text))]
