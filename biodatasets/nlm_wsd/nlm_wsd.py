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
In order to support research investigating the automatic resolution of word sense ambiguity using natural language
processing techniques, we have constructed this test collection of medical text in which the ambiguities were resolved
by hand. Evaluators were asked to examine instances of an ambiguous word and determine the sense intended by selecting
the Metathesaurus concept (if any) that best represents the meaning of that sense. The test collection consists of 50
highly frequent ambiguous UMLS concepts from 1998 MEDLINE. Each of the 50 ambiguous cases has 100 ambiguous instances
randomly selected from the 1998 MEDLINE citations. For a total of 5,000 instances. We had a total of 11 evaluators of
which 8 completed 100% of the 5,000 instances, 1 completed 56%, 1 completed 44%, and the final evaluator completed 12%
of the instances. Evaluations were only used when the evaluators completed all 100 instances for a given ambiguity.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict

import datasets
from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

_CITATION = """\
@article{,
  title    = "Developing a test collection for biomedical word sense
              disambiguation",
  author   = "Weeber, M and Mork, J G and Aronson, A R",
  journal  = "Proc AMIA Symp",
  pages    = "746--750",
  year     =  2001,
  language = "en"
}
"""

_DATASETNAME = "nlm_wsd"

_DESCRIPTION = """\
In order to support research investigating the automatic resolution of word sense ambiguity using natural language
processing techniques, we have constructed this test collection of medical text in which the ambiguities were resolved
by hand. Evaluators were asked to examine instances of an ambiguous word and determine the sense intended by selecting
the Metathesaurus concept (if any) that best represents the meaning of that sense. The test collection consists of 50
highly frequent ambiguous UMLS concepts from 1998 MEDLINE. Each of the 50 ambiguous cases has 100 ambiguous instances
randomly selected from the 1998 MEDLINE citations. For a total of 5,000 instances. We had a total of 11 evaluators of
which 8 completed 100% of the 5,000 instances, 1 completed 56%, 1 completed 44%, and the final evaluator completed 12%
of the instances. Evaluations were only used when the evaluators completed all 100 instances for a given ambiguity.
"""

_HOMEPAGE = "https://lhncbc.nlm.nih.gov/restricted/ii/areas/WSD/index.html"

_LICENSE = "DUA (UMLS)"

_URLS = {}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_DISAMBIGUATION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"

class NlmWsdDataset(datasets.GeneratorBasedBuilder):
    """Biomedical Word Sense Disambiguation (WSD)."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="nlm_wsd_source",
            version=SOURCE_VERSION,
            description="NLM-WSD source schema",
            schema="source",
            subset_id="nlm_wsd_",
        ),
        BigBioConfig(
            name="nlm_wsd_bigbio_kb",
            version=BIGBIO_VERSION,
            description="NLM-WSD BigBio schema",
            schema="bigbio_kb",
            subset_id="nlm_wsd",
        ),
    ]
    """
    BUILDER_CONFIGS = [
        BigBioConfig(
            name="nlm_wsd_non_reviewed_source",
            version=SOURCE_VERSION,
            description="NLM-WSD basic non reviewed source schema",
            schema="source",
            subset_id="nlm_wsd_non_reviewed",
        ),
        BigBioConfig(
            name="nlm_wsd_non_reviewed_bigbio_kb",
            version=BIGBIO_VERSION,
            description="NLM-WSD basic non reviewed BigBio schema",
            schema="bigbio_kb",
            subset_id="nlm_wsd_non_reviewed",
        ),
        BigBioConfig(
            name="nlm_wsd_reviewed_source",
            version=SOURCE_VERSION,
            description="NLM-WSD basic reviewed source schema",
            schema="source",
            subset_id="nlm_wsd_reviewed",
        ),
        BigBioConfig(
            name="nlm_wsd_reviewed_bigbio_kb",
            version=BIGBIO_VERSION,
            description="NLM-WSD basic reviewed BigBio schema",
            schema="bigbio_kb",
            subset_id="nlm_wsd_reviewed",
        ),
    ]
    """

    DEFAULT_CONFIG_NAME = "nlm_wsd_reviewed_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "sentence_id": datasets.Value("string"),
                    "label": datasets.Value("string"),
                    "sentence": {
                        "text": datasets.Value("string"),
                        "ambiguous_word": datasets.Value("string"),
                        "ambiguous_word_alias": datasets.Value("string"),
                        "offsets_context": datasets.Sequence("int32"),
                        "offsets_ambiguity": datasets.Sequence("int32"),
                        "context": datasets.Value("string"),
                    },
                    "citation":{
                        "text": datasets.Value("string"),
                        "ambiguous_word": datasets.Value("string"),
                        "ambiguous_word_alias": datasets.Value("string"),
                        "offsets_context": datasets.Sequence("int32"),
                        "offsets_ambiguity": datasets.Sequence("int32"),
                        "context": datasets.Value("string"),
                    }
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
        
        if self.config.data_dir is None:
            raise ValueError("This is a local dataset. Please pass the data_dir kwarg to load_dataset.")
        else:
            data_dir = self.config.data_dir

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_dir": Path(data_dir),
                },
            )
        ]

    def _generate_examples(self, data_dir: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        
        for dir in data_dir.iterdir():
            if self.config.schema == "source":
                for key, example in self._generate_parsed_documents(dir):
                    yield key, example

            elif self.config.schema == "bigbio_kb":
                for key, example in self._generate_parsed_documents(dir):
                    yield key, example

    def _generate_parsed_documents(self, dir):
        file_path = dir / f"{dir.name}_set"
        with file_path.open() as f:
            for raw_document in self._generate_raw_documents(f):
                document = {}
                id, document_id, label = raw_document[0].split("|")

                info_sentence = self._parse_ambig_pos_info(raw_document[2])
                info_sentence["text"] = raw_document[1]

                info_citation = self._parse_ambig_pos_info(raw_document[-1])
                n_cit = len(raw_document) - 3
                info_citation["text"] = "\n".join(raw_document[3:3+n_cit])

                document = {
                    "id": id,
                    "document_id": document_id,
                    "label": label,
                    "sentence": info_sentence,
                    "citation": info_citation
                }
                yield document


    def _generate_raw_documents(self, fstream):
        raw_document = []
        for line in fstream:
            if line.strip():
                raw_document.append(line.strip())
            elif raw_document:
                yield raw_document
                raw_document = []
        # needed for last document
        if raw_document:
            yield raw_document

    def _parse_ambig_pos_info(self, line):
         infos = line.split("|")
         assert len(infos) == 7
         pos_info = {
             "amiguous_word": infos[0], 
             "ambiguous_word_alias": infos[1], 
             "offsets_context": [infos[2], infos[3]], 
             "offsets_ambiguity": [infos[4], infos[5]], 
             "context": infos[6]
         }
         return pos_info

