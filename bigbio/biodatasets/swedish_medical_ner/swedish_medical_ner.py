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
swedish_medical_ner is Named Entity Recognition dataset on medical text in Swedish. 

It consists three subsets which are in turn derived from three different sources respectively: 

* the Swedish Wikipedia (a.k.a. wiki): Wiki_annotated_60.txt
* Läkartidningen (a.k.a. lt): LT_annotated_60.txt
* 1177 Vårdguiden (a.k.a. 1177): 1177_annotated_sentences.txt

Texts from both Swedish Wikipedia and Läkartidningen were automatically annotated using a 
list of medical seed terms. Sentences from 1177 Vårdguiden were manuually annotated.

It can be found in Hugging Face Datasets: https://huggingface.co/datasets/swedish_medical_ner.

"""

import os
import re
from typing import Dict, List, Tuple

import datasets

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks

_DATASETNAME = "swedish_medical_ner"

_LANGUAGES = [Lang.SV]
_LOCAL = False
_CITATION = """\
@inproceedings{almgren-etal-2016-named,
    author = {
        Almgren, Simon and
        Pavlov, Sean and
        Mogren, Olof
    },
    title     = {Named Entity Recognition in Swedish Medical Journals with Deep Bidirectional Character-Based LSTMs},
    booktitle = {Proceedings of the Fifth Workshop on Building and Evaluating Resources for Biomedical Text Mining (BioTxtM 2016)},
    publisher = {The COLING 2016 Organizing Committee},
    pages     = {30-39},
    year      = {2016},
    month     = {12},
    url       = {https://aclanthology.org/W16-5104},
    eprint    = {https://aclanthology.org/W16-5104.pdf}
}
"""

_DESCRIPTION = """\
swedish_medical_ner is Named Entity Recognition dataset on medical text in Swedish. 
It consists three subsets which are in turn derived from three different sources 
respectively: the Swedish Wikipedia (a.k.a. wiki), Läkartidningen (a.k.a. lt), 
and 1177 Vårdguiden (a.k.a. 1177). While the Swedish Wikipedia and Läkartidningen 
subsets in total contains over 790000 sequences with 60 characters each, 
the 1177 Vårdguiden subset is manually annotated and contains 927 sentences, 
2740 annotations, out of which 1574 are disorder and findings, 546 are 
pharmaceutical drug, and 620 are body structure.

Texts from both Swedish Wikipedia and Läkartidningen were automatically annotated 
using a list of medical seed terms. Sentences from 1177 Vårdguiden were manuually 
annotated.
"""

_HOMEPAGE = "https://github.com/olofmogren/biomedical-ner-data-swedish/"

_LICENSE = "Creative Commons Attribution-ShareAlike 4.0 International Public License (CC BY-SA 4.0)"

_URLS = {
    "swedish_medical_ner_wiki": "https://raw.githubusercontent.com/olofmogren/biomedical-ner-data-swedish/master/Wiki_annotated_60.txt",
    "swedish_medical_ner_lt": "https://raw.githubusercontent.com/olofmogren/biomedical-ner-data-swedish/master/LT_annotated_60.txt",
    "swedish_medical_ner_1177": "https://raw.githubusercontent.com/olofmogren/biomedical-ner-data-swedish/master/1177_annotated_sentences.txt",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class SwedishMedicalNerDataset(datasets.GeneratorBasedBuilder):
    """
    Swedish medical named entity recognition

    The dataset contains three subsets, namely "wiki", "lt" and "1177".
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = []
    for subset in ["wiki", "lt", "1177"]:
        BUILDER_CONFIGS.append(
            BigBioConfig(
                name=f"swedish_medical_ner_{subset}_source",
                version=SOURCE_VERSION,
                description="swedish_medical_ner source schema",
                schema="source",
                subset_id=f"swedish_medical_ner_{subset}",
            )
        )
        BUILDER_CONFIGS.append(
            BigBioConfig(
                name=f"swedish_medical_ner_{subset}_bigbio_kb",
                version=BIGBIO_VERSION,
                description="swedish_medical_ner BigBio schema",
                schema="bigbio_kb",
                subset_id=f"swedish_medical_ner_{subset}",
            )
        )

    DEFAULT_CONFIG_NAME = "swedish_medical_ner_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "sid": datasets.Value("string"),
                    "sentence": datasets.Value("string"),
                    "entities": [
                        {
                            "start": datasets.Value("int32"),
                            "end": datasets.Value("int32"),
                            "text": datasets.Value("string"),
                            "type": datasets.Value("string"),
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

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        urls = _URLS
        filepath = dl_manager.download_and_extract(urls[self.config.subset_id])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": filepath,
                    "split": "train",
                },
            ),
        ]

    @staticmethod
    def get_type(text):
        """
        Tagging format per the dataset authors
        - Prenthesis, (): Disorder and Finding
        - Brackets, []: Pharmaceutical Drug
        - Curly brackets, {}: Body Structure

        """
        if text[0] == "(":
            return "disorder_finding"
        elif text[1] == "[":
            return "pharma_drug"
        return "body_structure"

    @staticmethod
    def get_source_example(uid, tagged):

        ents, text = zip(*tagged)
        text = list(text)

        # build offsets
        offsets = []
        curr = 0
        for span in text:
            offsets.append((curr, curr + len(span)))
            curr = curr + len(span)

        text = "".join(text)
        doc = {"sid": "s" + str(uid), "sentence": text, "entities": []}

        # Create entities
        for i, (start, end) in enumerate(offsets):
            if ents[i] is not None:
                doc["entities"].append(
                    {
                        "start": start,
                        "end": end,
                        "text": text[start:end],
                        "type": ents[i],
                    }
                )

        return uid, doc

    @staticmethod
    def get_bigbio_example(uid, tagged, remove_markup=True):
        doc = {
            "id": str(uid),
            "document_id": "s" + str(uid),
            "passages": [],
            "entities": [],
            "events": [],
            "coreferences": [],
            "relations": [],
        }

        ents, text = zip(*tagged)
        text = list(text)
        if remove_markup:
            for i in range(len(ents)):
                if ents[i] is not None:
                    text[i] = re.sub(r"[(){}\[\]]", "", text[i]).strip()

        # build offsets
        offsets = []
        curr = 0
        for span in text:
            offsets.append((curr, curr + len(span)))
            curr = curr + len(span)

        # Create passage
        passage = "".join(text)
        doc["passages"].append(
            {
                "id": str(uid) + "-passage-0",
                "type": "sentence",
                "text": [passage],
                "offsets": [[0, len(passage)]],
            }
        )

        # Create entities
        ii = 0
        for i, (start, end) in enumerate(offsets):
            if ents[i] is not None:
                doc["entities"].append(
                    {
                        "id": str(uid) + "-entity-" + str(ii),
                        "type": ents[i],
                        "text": [passage[start:end]],
                        "offsets": [[start, end]],
                        "normalized": [],
                    }
                )
                ii += 1

        return uid, doc

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        entity_rgx = re.compile(r"[(].+?[)]|[\[].+?[\]]|[{].+?[}]")

        uid = 0
        with open(filepath, "rt", encoding="utf-8") as file:
            for i, row in enumerate(file):
                row = row.replace("\n", "")
                if row:
                    curr = 0
                    stack = []
                    # match entities and build spans for sentence string
                    for m in entity_rgx.finditer(row):
                        span = m.group()
                        if m.start() != 0:
                            stack.append([None, row[curr : m.start()]])
                        stack.append((self.get_type(span), span))
                        curr = m.start() + len(span)
                    stack.append([None, row[curr:]])

                    if self.config.schema == "source":
                        yield self.get_source_example(uid, stack)
                    elif self.config.schema == "bigbio_kb":
                        yield self.get_bigbio_example(uid, stack)
                    uid += 1
