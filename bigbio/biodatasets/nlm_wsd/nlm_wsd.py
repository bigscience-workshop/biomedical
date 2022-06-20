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

Comment from author:
BigBio schema fixes off by one error of end offset of entities. The source config remains unchanged.

Instructions on how to load locally:
1) Create directory
2) Download one of the following annotation sets from https://lhncbc.nlm.nih.gov/restricted/ii/areas/WSD/index.html
   and put it into the folder:
   - Full Reviewed Set
     https://lhncbc.nlm.nih.gov/restricted/ii/areas/WSD/downloads/full_reviewed_results.tar.gz
     (Link "Full Reviewed Result Set (requires Common Files above)")
     subset_id = nlm_wsd_reviewed
   - Full Non-Reviewed Set
     https://lhncbc.nlm.nih.gov/restricted/ii/areas/WSD/downloads/full_non_reviewed_results.tar.gz
     (Link "Full Non-Reviewed Result Set (requires Common Files above)")
     subset_id = nlm_wsd_non_reviewed
3) Download https://lhncbc.nlm.nih.gov/restricted/ii/areas/WSD/downloads/UMLS1999.tar.gz (Link "1999 UMLS Data Files")
   and put it into the folder
4) Set kwarg data_dir of load_datasets to the path of the directory
"""

import itertools as it
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tags, Tasks
from bigbio.utils.license import Licenses

_TAGS = [Tags.ABBREVIATION]
_LANGUAGES = [Lang.EN]
_PUBMED = True
_LOCAL = True
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

_LICENSE = Licenses.UMLS_LICENSE

_URLS = {
    "UMLS": "UMLS1999.tar.gz",
    "reviewed": "full_reviewed_results.tar.gz",
    "non_reviewed": "full_non_reviewed_results.tar.gz",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_DISAMBIGUATION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


@dataclass
class NlmWsdBigBioConfig(BigBioConfig):
    schema: str = "source"
    name: str = "nlm_wsd_reviewed_source"
    version: datasets.Version = datasets.Version(_SOURCE_VERSION)
    description: str = "NLM-WSD basic reviewed source schema"
    subset_id: str = "nlm_wsd_reviewed"


class NlmWsdDataset(datasets.GeneratorBasedBuilder):
    """Biomedical Word Sense Disambiguation (WSD)."""

    uid = it.count(0)

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        NlmWsdBigBioConfig(
            name="nlm_wsd_non_reviewed_source",
            version=SOURCE_VERSION,
            description="NLM-WSD basic non reviewed source schema",
            schema="source",
            subset_id="nlm_wsd_non_reviewed",
        ),
        NlmWsdBigBioConfig(
            name="nlm_wsd_non_reviewed_bigbio_kb",
            version=BIGBIO_VERSION,
            description="NLM-WSD basic non reviewed BigBio schema",
            schema="bigbio_kb",
            subset_id="nlm_wsd_non_reviewed",
        ),
        NlmWsdBigBioConfig(
            name="nlm_wsd_reviewed_source",
            version=SOURCE_VERSION,
            description="NLM-WSD basic reviewed source schema",
            schema="source",
            subset_id="nlm_wsd_reviewed",
        ),
        NlmWsdBigBioConfig(
            name="nlm_wsd_reviewed_bigbio_kb",
            version=BIGBIO_VERSION,
            description="NLM-WSD basic reviewed BigBio schema",
            schema="bigbio_kb",
            subset_id="nlm_wsd_reviewed",
        ),
    ]

    BUILDER_CONFIG_CLASS = NlmWsdBigBioConfig

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
                        "offsets_context": datasets.Sequence(datasets.Value("int32")),
                        "offsets_ambiguity": datasets.Sequence(datasets.Value("int32")),
                        "context": datasets.Value("string"),
                    },
                    "citation": {
                        "text": datasets.Value("string"),
                        "ambiguous_word": datasets.Value("string"),
                        "ambiguous_word_alias": datasets.Value("string"),
                        "offsets_context": datasets.Sequence(datasets.Value("int32")),
                        "offsets_ambiguity": datasets.Sequence(datasets.Value("int32")),
                        "context": datasets.Value("string"),
                    },
                    "choices": [
                        {
                            "label": datasets.Value("string"),
                            "concept": datasets.Value("string"),
                            "cui": datasets.Value("string"),
                            "type": [datasets.Value("string")],
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

        if self.config.data_dir is None:
            raise ValueError(
                "This is a local dataset. Please pass the data_dir kwarg to load_dataset."
            )
        else:
            data_dir = Path(self.config.data_dir)
            umls_dir = dl_manager.download_and_extract(data_dir / _URLS["UMLS"])
            mrcon_path = Path(umls_dir) / "META" / "MRCON"
            if self.config.subset_id == "nlm_wsd_reviewed":
                ann_dir = dl_manager.download_and_extract(data_dir / _URLS["reviewed"])
                ann_dir = Path(ann_dir) / "Reviewed_Results"
            else:
                ann_dir = dl_manager.download_and_extract(
                    data_dir / _URLS["non_reviewed"]
                )
                ann_dir = Path(ann_dir) / "Non-Reviewed_Results"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "mrcon_path": mrcon_path,
                    "ann_dir": ann_dir,
                },
            )
        ]

    def _generate_examples(self, mrcon_path: Path, ann_dir: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        # read label->cui map
        umls_map = {}
        with mrcon_path.open() as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        for line in content:
            fields = line.split("|")
            assert len(fields) == 9, f"{len(fields)}"
            assert fields[0][0] == "C"
            umls_map[fields[6]] = fields[0]

        for dir in ann_dir.iterdir():
            if self.config.schema == "source" and dir.is_dir():
                for example in self._generate_parsed_documents(dir, umls_map):
                    yield next(self.uid), example

            elif self.config.schema == "bigbio_kb" and dir.is_dir():
                for example in self._generate_parsed_documents(dir, umls_map):
                    yield next(self.uid), self._source_to_kb(example)

    def _generate_parsed_documents(self, dir, umls_map):

        # read choices
        choices = []
        choices_path = dir / "choices"
        with choices_path.open() as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        for line in content:
            label, concept, *type = line.split("|")
            type = [x.split(", ")[1] for x in type]
            m = re.search(r"(?<=\().+(?=\))", concept)
            if m is None:
                choices.append(
                    {"label": label, "concept": concept, "type": type, "cui": ""}
                )
            else:
                concept = m.group()
                choices.append(
                    {
                        "label": label,
                        "concept": concept,
                        "type": type,
                        "cui": umls_map[concept],
                    }
                )

        file_path = dir / f"{dir.name}_set"
        with file_path.open() as f:
            for raw_document in self._generate_raw_documents(f):
                document = {}
                id, document_id, label = raw_document[0].strip().split("|")

                info_sentence = self._parse_ambig_pos_info(raw_document[2].strip())
                info_sentence["text"] = raw_document[1]

                info_citation = self._parse_ambig_pos_info(raw_document[-1].strip())
                n_cit = len(raw_document) - 3
                info_citation["text"] = "".join(raw_document[3 : 3 + n_cit])

                document = {
                    "id": id,
                    "sentence_id": document_id,
                    "label": label,
                    "sentence": info_sentence,
                    "citation": info_citation,
                    "choices": choices,
                }
                yield document

    def _generate_raw_documents(self, fstream):
        raw_document = []
        for line in fstream:
            if line.strip():
                raw_document.append(line)
            elif raw_document:
                yield raw_document
                raw_document = []
        # needed for last document
        if raw_document:
            yield raw_document

    def _parse_ambig_pos_info(self, line):
        infos = line.split("|")
        assert len(infos) == 8, f"{len(infos)}"
        pos_info = {
            "ambiguous_word": infos[0],
            "ambiguous_word_alias": infos[1],
            "offsets_context": [infos[2], infos[3]],
            "offsets_ambiguity": [infos[4], infos[5]],
            "context": infos[6],
        }
        return pos_info

    def _source_to_kb(self, example):
        document_ = {}
        document_["events"] = []
        document_["relations"] = []
        document_["coreferences"] = []
        document_["id"] = next(self.uid)
        document_["document_id"] = example["sentence_id"].split(".")[0]

        citation = example["citation"]
        document_["passages"] = [
            {
                "id": next(self.uid),
                "type": "",
                "text": [citation["text"]],
                "offsets": [[0, len(citation["text"])]],
            }
        ]
        choices = {x["label"]: x["cui"] for x in example["choices"]}
        types = {x["label"]: x["type"][0] for x in example["choices"]}

        db_id = (
            "" if example["label"] in ["None", "UNDEF"] else choices[example["label"]]
        )
        type = "" if example["label"] in ["None", "UNDEF"] else types[example["label"]]
        document_["entities"] = [
            {
                "id": next(self.uid),
                "type": type,
                "text": [citation["ambiguous_word_alias"]],
                "offsets": [
                    [
                        int(citation["offsets_ambiguity"][0]),
                        int(citation["offsets_ambiguity"][1]) + 1,
                    ]
                ],
                "normalized": [{"db_name": "UMLS", "db_id": db_id}],
            }
        ]
        return document_
