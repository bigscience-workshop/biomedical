# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and Simon Ott, github: nomisto
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
MedMentions is a new manually annotated resource for the recognition of biomedical concepts.
What distinguishes MedMentions from other annotated biomedical corpora is its size (over 4,000
abstracts and over 350,000 linked mentions), as well as the size of the concept ontology (over
3 million concepts from UMLS 2017) and its broad coverage of biomedical disciplines.

Corpus: The MedMentions corpus consists of 4,392 papers (Titles and Abstracts) randomly selected
from among papers released on PubMed in 2016, that were in the biomedical field, published in
the English language, and had both a Title and an Abstract.

Annotators: We recruited a team of professional annotators with rich experience in biomedical
content curation to exhaustively annotate all UMLS® (2017AA full version) entity mentions in
these papers.

Annotation quality: We did not collect stringent IAA (Inter-annotator agreement) data. To gain
insight on the annotation quality of MedMentions, we randomly selected eight papers from the
annotated corpus, containing a total of 469 concepts. Two biologists ('Reviewer') who did not
participate in the annotation task then each reviewed four papers. The agreement between
Reviewers and Annotators, an estimate of the Precision of the annotations, was 97.3%.

For more information visit: https://github.com/chanzuckerberg/MedMentions
"""

import itertools as it
from typing import List

import datasets

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks
from bigbio.utils.license import Licenses

_LANGUAGES = [Lang.EN]
_PUBMED = True
_LOCAL = False
_CITATION = """\
@misc{mohan2019medmentions,
      title={MedMentions: A Large Biomedical Corpus Annotated with UMLS Concepts},
      author={Sunil Mohan and Donghui Li},
      year={2019},
      eprint={1902.09476},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DATASETNAME = "medmentions"
_DISPLAYNAME = "MedMentions"

_DESCRIPTION = """\
MedMentions is a new manually annotated resource for the recognition of biomedical concepts.
What distinguishes MedMentions from other annotated biomedical corpora is its size (over 4,000
abstracts and over 350,000 linked mentions), as well as the size of the concept ontology (over
3 million concepts from UMLS 2017) and its broad coverage of biomedical disciplines.

Corpus: The MedMentions corpus consists of 4,392 papers (Titles and Abstracts) randomly selected
from among papers released on PubMed in 2016, that were in the biomedical field, published in
the English language, and had both a Title and an Abstract.

Annotators: We recruited a team of professional annotators with rich experience in biomedical
content curation to exhaustively annotate all UMLS® (2017AA full version) entity mentions in
these papers.

Annotation quality: We did not collect stringent IAA (Inter-annotator agreement) data. To gain
insight on the annotation quality of MedMentions, we randomly selected eight papers from the
annotated corpus, containing a total of 469 concepts. Two biologists ('Reviewer') who did not
participate in the annotation task then each reviewed four papers. The agreement between
Reviewers and Annotators, an estimate of the Precision of the annotations, was 97.3%.
"""

_HOMEPAGE = "https://github.com/chanzuckerberg/MedMentions"

_LICENSE = Licenses.CC0_1p0

_URLS = {
    "medmentions_full": [
        "https://github.com/chanzuckerberg/MedMentions/raw/master/full/data/corpus_pubtator.txt.gz",
        "https://github.com/chanzuckerberg/MedMentions/raw/master/full/data/corpus_pubtator_pmids_trng.txt",
        "https://github.com/chanzuckerberg/MedMentions/raw/master/full/data/corpus_pubtator_pmids_dev.txt",
        "https://github.com/chanzuckerberg/MedMentions/raw/master/full/data/corpus_pubtator_pmids_test.txt",
    ],
    "medmentions_st21pv": [
        "https://github.com/chanzuckerberg/MedMentions/raw/master/st21pv/data/corpus_pubtator.txt.gz",
        "https://github.com/chanzuckerberg/MedMentions/raw/master/full/data/corpus_pubtator_pmids_trng.txt",
        "https://github.com/chanzuckerberg/MedMentions/raw/master/full/data/corpus_pubtator_pmids_dev.txt",
        "https://github.com/chanzuckerberg/MedMentions/raw/master/full/data/corpus_pubtator_pmids_test.txt",
    ],
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_DISAMBIGUATION, Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class MedMentionsDataset(datasets.GeneratorBasedBuilder):
    """MedMentions dataset for named-entity disambiguation (NED)"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="medmentions_full_source",
            version=SOURCE_VERSION,
            description="MedMentions Full source schema",
            schema="source",
            subset_id="medmentions_full",
        ),
        BigBioConfig(
            name="medmentions_full_bigbio_kb",
            version=BIGBIO_VERSION,
            description="MedMentions Full BigBio schema",
            schema="bigbio_kb",
            subset_id="medmentions_full",
        ),
        BigBioConfig(
            name="medmentions_st21pv_source",
            version=SOURCE_VERSION,
            description="MedMentions ST21pv source schema",
            schema="source",
            subset_id="medmentions_st21pv",
        ),
        BigBioConfig(
            name="medmentions_st21pv_bigbio_kb",
            version=BIGBIO_VERSION,
            description="MedMentions ST21pv BigBio schema",
            schema="bigbio_kb",
            subset_id="medmentions_st21pv",
        ),
    ]

    DEFAULT_CONFIG_NAME = "medmentions_full_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "pmid": datasets.Value("string"),
                    "passages": [
                        {
                            "type": datasets.Value("string"),
                            "text": datasets.Sequence(datasets.Value("string")),
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                        }
                    ],
                    "entities": [
                        {
                            "text": datasets.Sequence(datasets.Value("string")),
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                            "concept_id": datasets.Value("string"),
                            "semantic_type_id": datasets.Sequence(
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

        urls = _URLS[self.config.subset_id]
        (
            corpus_path,
            pmids_train,
            pmids_dev,
            pmids_test,
        ) = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"corpus_path": corpus_path, "pmids_path": pmids_train},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"corpus_path": corpus_path, "pmids_path": pmids_test},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"corpus_path": corpus_path, "pmids_path": pmids_dev},
            ),
        ]

    def _generate_examples(self, corpus_path, pmids_path):
        with open(pmids_path, encoding="utf8") as infile:
            pmids = infile.readlines()
        pmids = {int(x.strip()) for x in pmids}

        if self.config.schema == "source":
            with open(corpus_path, encoding="utf8") as corpus:
                for document in self._generate_parsed_documents(corpus, pmids):
                    yield document["pmid"], document

        elif self.config.schema == "bigbio_kb":
            uid = it.count(0)
            with open(corpus_path, encoding="utf8") as corpus:
                for document in self._generate_parsed_documents(corpus, pmids):
                    document["id"] = next(uid)
                    document["document_id"] = document.pop("pmid")

                    entities_ = []
                    for entity in document["entities"]:
                        for type in entity["semantic_type_id"]:
                            entities_.append(
                                {
                                    "id": next(uid),
                                    "type": type,
                                    "text": entity["text"],
                                    "offsets": entity["offsets"],
                                    "normalized": [
                                        {
                                            "db_name": "UMLS",
                                            "db_id": entity["concept_id"].split(':')[-1],
                                        }
                                    ],
                                }
                            )
                    document["entities"] = entities_

                    for passage in document["passages"]:
                        passage["id"] = next(uid)
                    document["relations"] = []
                    document["events"] = []
                    document["coreferences"] = []
                    yield document["document_id"], document

    def _generate_parsed_documents(self, fstream, pmids):
        for raw_document in self._generate_raw_documents(fstream):
            if self._parse_pmid(raw_document) in pmids:
                yield self._parse_document(raw_document)

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

    def _parse_pmid(self, raw_document):
        pmid, _ = raw_document[0].split("|", 1)
        return int(pmid)

    def _parse_document(self, raw_document):
        pmid, type, title = raw_document[0].split("|", 2)
        pmid_, type, abstract = raw_document[1].split("|", 2)
        passages = [
            {"type": "title", "text": [title], "offsets": [[0, len(title)]]},
            {
                "type": "abstract",
                "text": [abstract],
                "offsets": [[len(title) + 1, len(title) + len(abstract) + 1]],
            },
        ]

        entities = []
        for line in raw_document[2:]:
            (
                pmid_,
                start_idx,
                end_idx,
                mention,
                semantic_type_id,
                entity_id,
            ) = line.split("\t")
            entity = {
                "offsets": [[int(start_idx), int(end_idx)]],
                "text": [mention],
                "semantic_type_id": semantic_type_id.split(","),
                "concept_id": entity_id,
            }
            entities.append(entity)

        return {"pmid": int(pmid), "entities": entities, "passages": passages}
