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
A dataset loader for the SCAI Chemical dataset.

SCAI Chemical is a corpus of MEDLINE abstracts that has been annotated
to give an overview of the different chemical name classes
found in MEDLINE text.
"""

import gzip
from typing import Dict, List, Tuple

import datasets

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks
from bigbio.utils.license import Licenses

_LANGUAGES = [Lang.EN]
_PUBMED = True
_LOCAL = False
_CITATION = """\
@inproceedings{kolarik:lrec-ws08,
  author    = {Kol{\'a}{\vr}ik, Corinna and Klinger, Roman and Friedrich, Christoph M and Hofmann-Apitius, Martin and Fluck, Juliane},
  title     = {Chemical Names: {T}erminological Resources and Corpora Annotation},
  booktitle = {LREC Workshop on Building and Evaluating Resources for Biomedical Text Mining},
  year      = {2008},
}
"""

_DATASETNAME = "scai_chemical"

_DESCRIPTION = """\
SCAI Chemical is a corpus of MEDLINE abstracts that has been annotated
to give an overview of the different chemical name classes
found in MEDLINE text.
"""

_HOMEPAGE = "https://www.scai.fraunhofer.de/en/business-research-areas/bioinformatics/downloads/corpora-for-chemical-entity-recognition.html"

_LICENSE = Licenses.UNKNOWN

_URLS = {
    _DATASETNAME: "https://www.scai.fraunhofer.de/content/dam/scai/de/downloads/bioinformatik/Corpora-for-Chemical-Entity-Recognition/chemicals-test-corpus-27-04-2009-v3_iob.gz",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "3.0.0"

_BIGBIO_VERSION = "1.0.0"


class ScaiChemicalDataset(datasets.GeneratorBasedBuilder):
    """SCAI Chemical is a dataset annotated in 2008 with mentions of chemicals."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="scai_chemical_source",
            version=SOURCE_VERSION,
            description="SCAI Chemical source schema",
            schema="source",
            subset_id="scai_chemical",
        ),
        BigBioConfig(
            name="scai_chemical_bigbio_kb",
            version=BIGBIO_VERSION,
            description="SCAI Chemical BigBio schema",
            schema="bigbio_kb",
            subset_id="scai_chemical",
        ),
    ]

    DEFAULT_CONFIG_NAME = "scai_chemical_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "document_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "tokens": [
                        {
                            "offsets": [datasets.Value("int64")],
                            "text": datasets.Value("string"),
                            "tag": datasets.Value("string"),
                        }
                    ],
                    "entities": [
                        {
                            "offsets": [datasets.Value("int64")],
                            "text": datasets.Value("string"),
                            "type": datasets.Value("string"),
                        }
                    ],
                }
            )

        elif self.config.schema == "bigbio_kb":
            features = schemas.kb_features
        else:
            raise ValueError("Unrecognized schema: %s" % self.config.schema)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        url = _URLS[_DATASETNAME]
        filepath = dl_manager.download(url)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": filepath,
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        # Iterates through lines in file, collecting all lines belonging
        # to an example and converting into a single dict
        examples = []
        tokens = None
        with gzip.open(filepath, "rt", encoding="mac_roman") as data_file:
            print(filepath)
            for line in data_file:
                line = line.strip()
                if line.startswith("###"):
                    tokens = [line]
                elif line == "":
                    examples.append(self._make_example(tokens))
                else:
                    tokens.append(line)

        # Returns the examples using the desired schema
        if self.config.schema == "source":
            for i, example in enumerate(examples):
                yield i, example

        elif self.config.schema == "bigbio_kb":
            for i, example in enumerate(examples):
                bigbio_example = {
                    "id": "example-" + str(i),
                    "document_id": example["document_id"],
                    "passages": [
                        {
                            "id": "passage-" + str(i),
                            "type": "abstract",
                            "text": [example["text"]],
                            "offsets": [[0, len(example["text"])]],
                        }
                    ],
                    "entities": [],
                    "events": [],
                    "coreferences": [],
                    "relations": [],
                }

                # Converts entities to BigBio format
                for j, entity in enumerate(example["entities"]):
                    bigbio_example["entities"].append(
                        {
                            "id": "entity-" + str(i) + "-" + str(j),
                            "offsets": [entity["offsets"]],
                            "text": [entity["text"]],
                            "type": entity["type"],
                            "normalized": [],
                        }
                    )

                yield i, bigbio_example

    @staticmethod
    def _make_example(tokens):
        """
        Converts a list of lines representing tokens into an example dictionary
        formatted according to the source schema

        :param tokens: list of strings
        :return: dictionary in the source schema
        """
        document_id = tokens[0][4:]

        text = ""
        processed_tokens = []
        entities = []
        last_offset = 0

        for token in tokens[1:]:
            token_pieces = token.split("\t")
            if len(token_pieces) != 5:
                raise ValueError("Failed to parse line: %s" % token)

            token_text = str(token_pieces[0])
            token_start = int(token_pieces[1])
            token_end = int(token_pieces[2])
            entity_text = str(token_pieces[3])
            token_tag = str(token_pieces[4])[1:]

            if token_start > last_offset:
                for _ in range(token_start - last_offset):
                    text += " "
            elif token_start < last_offset:
                raise ValueError("Invalid start index: %s" % token)
            last_offset = token_end

            text += token_text
            processed_tokens.append(
                {
                    "offsets": [token_start, token_end],
                    "text": token_text,
                    "tag": token_tag,
                }
            )
            if entity_text != "":
                entities.append(
                    {
                        "offsets": [token_start, token_start + len(entity_text)],
                        "text": entity_text,
                        "type": token_tag[2:],
                    }
                )

        return {
            "document_id": document_id,
            "text": text,
            "entities": entities,
            "tokens": processed_tokens,
        }
