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
Evaluation of Word Sense Disambiguation methods (WSD) in the biomedical domain is difficult because the available
resources are either too small or too focused on specific types of entities (e.g. diseases or genes). We have
developed a method that can be used to automatically develop a WSD test collection using the Unified Medical Language
System (UMLS) Metathesaurus and the manual MeSH indexing of MEDLINE. The resulting dataset is called MSH WSD and
consists of 106 ambiguous abbreviations, 88 ambiguous terms and 9 which are a combination of both, for a total of 203
ambiguous words. Each instance containing the ambiguous word was assigned a CUI from the 2009AB version of the UMLS.
For each ambiguous term/abbreviation, the data set contains a maximum of 100 instances per sense obtained from
MEDLINE; totaling 37,888 ambiguity cases in 37,090 MEDLINE citations.
"""

# Note from the Author how to load dataset:
# 1) Download the file MSHCorpus.zip (Link "MSHWSD Data Set") from  https://lhncbc.nlm.nih.gov/ii/areas/WSD/collaboration.html
# 2) Unzip MSHCorpus.zip
# 3) Set kwarg data_dir to the path of the folder containing the .arrf files ("MSHCorpus")

import itertools as it
import re
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

_CITATION = """\
@article{,
    author={Jimeno-Yepes, Antonio J.
    and McInnes, Bridget T.
    and Aronson, Alan R.},
    title={Exploiting MeSH indexing in MEDLINE to generate a data set for word sense disambiguation},
    journal={BMC Bioinformatics},
    year={2011},
    month={Jun},
    day={02},
    volume={12},
    number={1},
    pages={223},
    issn={1471-2105},
    doi={10.1186/1471-2105-12-223},
    url={https://doi.org/10.1186/1471-2105-12-223}
}

"""

_DATASETNAME = "msh_wsd"

_DESCRIPTION = """\
Evaluation of Word Sense Disambiguation methods (WSD) in the biomedical domain is difficult because the available
resources are either too small or too focused on specific types of entities (e.g. diseases or genes). We have
developed a method that can be used to automatically develop a WSD test collection using the Unified Medical Language
System (UMLS) Metathesaurus and the manual MeSH indexing of MEDLINE. The resulting dataset is called MSH WSD and
consists of 106 ambiguous abbreviations, 88 ambiguous terms and 9 which are a combination of both, for a total of 203
ambiguous words. Each instance containing the ambiguous word was assigned a CUI from the 2009AB version of the UMLS.
For each ambiguous term/abbreviation, the data set contains a maximum of 100 instances per sense obtained from
MEDLINE; totaling 37,888 ambiguity cases in 37,090 MEDLINE citations.
"""

_HOMEPAGE = "https://lhncbc.nlm.nih.gov/ii/areas/WSD/collaboration.html"

_LICENSE = "DUA (UMLS)"

_URLS = {
    _DATASETNAME: ""
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_DISAMBIGUATION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class MshWsdDataset(datasets.GeneratorBasedBuilder):
    """Biomedical Word Sense Disambiguation (WSD)."""

    uid = it.count(0)

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="msh_wsd_source",
            version=SOURCE_VERSION,
            description="MSH-WSD source schema",
            schema="source",
            subset_id="msh_wsd",
        ),
        BigBioConfig(
            name="msh_wsd_bigbio_kb",
            version=BIGBIO_VERSION,
            description="MSH-WSD BigBio schema",
            schema="bigbio_kb",
            subset_id="msh_wsd",
        ),
    ]

    DEFAULT_CONFIG_NAME = "msh_wsd_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "ambiguous_word": datasets.Value("string"),
                    "sentences": [
                        {
                            "pmid": datasets.Value("string"),
                            "text": datasets.Value("string"),
                            "label": datasets.Value("string"),
                        }
                    ],
                    "choices": [
                        {
                            "label": datasets.Value("string"),
                            "concept": datasets.Value("string"),
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
            ),
        ]

    def _generate_examples(self, data_dir: Path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        concepts = data_dir / "benchmark_mesh.txt"
        with concepts.open() as f:
            concepts = f.readlines()
        concepts = [x.strip().split("\t") for x in concepts]

        concept_map = {cuis[0]: {f"M{num}": cui for num, cui in enumerate(cuis[1:], 1)} for cuis in concepts}

        files = list(data_dir.glob("*arff"))
        for guid, file in enumerate(files):
            if self.config.schema == "source":
                for example in self._parse_document(concept_map, file):
                    yield guid, example

            elif self.config.schema == "bigbio_kb":
                for document in self._parse_document(concept_map, file):
                    for example in self._source_to_kb(document):
                        yield example["id"], example

    def _parse_document(self, concept_map, file: Path):

        with file.open() as f:
            content = f.readlines()
        content = [x.strip() for x in content]

        # search line number of @DATA, sometimes 6 or 7
        start_l = None
        for number, line in enumerate(content):
            if line.startswith("@DATA"):
                start_l = number + 1
                break
        assert start_l is not None

        amb_word = file.with_suffix("").name[: -len("_pmids_tagged")]

        sentences = []
        for line in content[start_l:]:
            # cant use , or ," ", as seperator
            m_pmid = re.search("[0-9]+(?=(,))", line)
            pmid = m_pmid.group()
            m_label = re.search("(?<=(,))M[0-9]+", line)
            label = m_label.group()

            citation = line[m_pmid.span()[1] + 1 : m_label.span()[0] - 1].strip('"')

            sentences.append({"pmid": pmid, "text": citation, "label": label})

        yield {
            "ambiguous_word": amb_word,
            "sentences": sentences,
            "choices": [{"label": key, "concept": value} for key, value in concept_map[amb_word].items()],
        }

    def _source_to_kb(self, document):
        choices = {x["label"]: x["concept"] for x in document["choices"]}
        for sentence in document["sentences"]:
            document_ = {}
            document_["events"] = []
            document_["relations"] = []
            document_["coreferences"] = []
            document_["id"] = next(self.uid)
            document_["document_id"] = sentence["pmid"]
            document_["passages"] = [
                {"id": next(self.uid), "type": "", "text": [sentence["text"]], "offsets": [[0, len(sentence["text"])]]}
            ]
            document_["entities"] = [
                {
                    "id": next(self.uid),
                    "type": "",
                    "text": [document["ambiguous_word"]],
                    "offsets": [self._parse_offset(sentence["text"])],
                    "normalized": [{"db_name": "MeSH", "db_id": choices[sentence["label"]]}],
                }
            ]
            yield document_

    def _parse_offset(self, sentence):
        m = re.search("(?<=(<e>)).+(?=(</e>))", sentence)
        return m.span()
