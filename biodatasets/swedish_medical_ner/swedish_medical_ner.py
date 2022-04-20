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
from typing import Dict, List, Tuple

import datasets
import regex as re

from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

_DATASETNAME = "swedish_medical_ner"

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
    _DATASETNAME: [
        "https://raw.githubusercontent.com/olofmogren/biomedical-ner-data-swedish/master/Wiki_annotated_60.txt",
        "https://raw.githubusercontent.com/olofmogren/biomedical-ner-data-swedish/master/LT_annotated_60.txt",
        "https://raw.githubusercontent.com/olofmogren/biomedical-ner-data-swedish/master/1177_annotated_sentences.txt",
    ],
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


def _read_data(datapaths):
    """
    This function does the following:
    1) Loads all three sub data sets;
    2) Preprocesses original sentences to get rid of brackets;
    3) Extracts entities, entity types and their char offsets;
    4) Returns data dict per sample/sentence.
    """

    def find_type(s, e):
        """
        Matching entity type from brackets/braces/parentheses.
        :param s: left/open bracket/brace/parenthese
        :param e: right/closed left bracket/brace/parenthese
        """
        if (s == "(") and (e == ")"):
            return "Disorder and Finding"
        elif (s == "[") and (e == "]"):
            return "Pharmaceutical Drug"
        elif (s == "{") and (e == "}"):
            return "Body Structure"
        else:
            return ""

    _ents = r"{[^{}]+}|\[[^\[\]]+\]|\([^\(\)]+\)"
    _spaces = r"(?<=[\[{(])\s+|\s+(?=[\]})])"
    documents = []
    s_id = 0
    for filepath in datapaths:
        db_id = 1
        if db_id == 1:
            db_name = "wiki"
        elif db_id == 2:
            db_name = "LT"
        else:
            db_name = "1177"

        with open(filepath, encoding="utf-8") as f:
            for _, row in enumerate(f):
                sentence = row.replace("\n", "")
                # Remove extra white spaces within the brackets:
                sentence = re.sub(_spaces, "", sentence)
                # In the original data, entities are wrapped with brackets/braces/parentheses.
                # Here we identify entities with brackets:
                matches = re.findall(_ents, sentence, overlapped=True)
                # Remove brackets/braces/parentheses
                words = [re.sub(r"[{}\[\]\(\)]", "", m) for m in matches]
                # Match entity types
                types = [find_type(match[0], match[-1]) for match in matches]
                # Add space to the entity and the word on its left (if there isn't a space in between), e.g.
                # "jukdom(KOL)(lungfunktionsundersökning)" -> "jukdom (KOL) (lungfunktionsundersökning)"
                clean_sent = re.sub(r"(\w|\)|\]|})([\(\[{])", "\\1 \\2", sentence)
                # Add space to the entity and the word on its right (if there isn't a space in between), e.g.
                # "(kronisk bronkit)och" -> "(kronisk bronkit) och"
                clean_sent = re.sub(r"([\)\]}])(\w|\(|\[|{)", "\\1 \\2", clean_sent)
                # Remove the brackets/braces/parentheses around each entity
                clean_sent = re.sub(r"[{}\[\]\(\)]", "", clean_sent)

                # Remove brackets/braces/parentheses from sentences and find the char offsets of each entity
                # in the cleaned sentence
                try:
                    targets = [
                        {
                            "start": m.start(2),
                            "end": m.end(2),
                            "text": clean_sent[m.start(2) : m.end(2)],
                        }
                        for m in map(lambda word: re.search(f"(^|[^\w]+)({word})($|[^\w]+)", clean_sent), words)
                    ]
                # Skip wrongly formatted entities
                # e.g. "rån patienter med (Barretts ){e)s)o)f)a)g)u)s),} påverkas av korta pulse"
                # e.g. "let samt (imperforerad ){a)n)u)s).} Möten mellan himmel och jord"
                # I have decided to ignore these cases since they are difficult to fix and would require the use
                # of regex thus increase processing time.
                except:
                    continue

                if targets:
                    documents.append(
                        {
                            "sid": "s" + str(s_id),
                            "sentence": clean_sent,
                            "entities": [
                                {
                                    "start": targets[i]["start"],
                                    "end": targets[i]["end"],
                                    "text": targets[i]["text"],
                                    "type": types[i],
                                }
                                for i in range(len(targets))
                            ],
                            "normalized": {"db_name": db_name, "db_id": "d" + str(db_id)},
                        }
                    )
                s_id += 1
        db_id += 1

    return documents


class SwedishMedicalNerDataset(datasets.GeneratorBasedBuilder):
    """Swedish medical named entity recognition"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="swedish_medical_ner_source",
            version=SOURCE_VERSION,
            description="swedish_medical_ner source schema",
            schema="source",
            subset_id="swedish_medical_ner",
        ),
        BigBioConfig(
            name="swedish_medical_ner_bigbio_kb",
            version=BIGBIO_VERSION,
            description="swedish_medical_ner BigBio schema",
            schema="bigbio_kb",
            subset_id="swedish_medical_ner",
        ),
    ]

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

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_dir": data_dir,
                    "split": "train",
                },
            ),
        ]

    @staticmethod
    def _get_source_sample(sample):
        return {"sid": sample["sid"], "sentence": sample["sentence"], "entities": sample["entities"]}

    @staticmethod
    def _get_bigbio_sample(sample_id, sample):
        return {
            "id": str(sample_id),
            "document_id": sample["sid"],
            "passages": [
                {
                    "id": sample["sid"] + "-passage-0",
                    "type": "sentence",
                    "text": [sample["sentence"]],
                    "offsets": [[0, len(sample["sentence"])]],
                }
            ],
            "entities": [
                {
                    "id": sample["sid"] + "-entity-" + str(i),
                    "text": [ent["text"]],
                    "offsets": [[ent["start"], ent["end"]]],
                    "type": ent["type"],
                    "normalized": [sample["normalized"]],
                }
                for i, ent in enumerate(sample["entities"])
            ],
            "events": [],
            "coreferences": [],
            "relations": [],
        }

    def _generate_examples(self, data_dir, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        _id = 0
        samples = _read_data(data_dir)
        for sample in samples:
            if self.config.schema == "source":
                yield _id, self._get_source_sample(sample)

            elif self.config.schema == "bigbio_kb":
                yield _id, self._get_bigbio_sample(_id, sample)
            _id += 1
