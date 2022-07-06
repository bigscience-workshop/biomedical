# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
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

import xml.etree.ElementTree as ET
from typing import Dict, Iterator, List, Tuple

import datasets

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks
from bigbio.utils.license import Licenses

_LANGUAGES = [Lang.EN]
_PUBMED = True
_LOCAL = False
_CITATION = """\
@Article{Bagewadi2014,
author={Bagewadi, Shweta
and Bobi{\'{c}}, Tamara
and Hofmann-Apitius, Martin
and Fluck, Juliane
and Klinger, Roman},
title={Detecting miRNA Mentions and Relations in Biomedical Literature},
journal={F1000Research},
year={2014},
month={Aug},
day={28},
publisher={F1000Research},
volume={3},
pages={205-205},
keywords={MicroRNAs; corpus; prediction algorithms},
abstract={
    INTRODUCTION: MicroRNAs (miRNAs) have demonstrated their potential as post-transcriptional
    gene expression regulators, participating in a wide spectrum of regulatory events such as
    apoptosis, differentiation, and stress response. Apart from the role of miRNAs in normal
    physiology, their dysregulation is implicated in a vast array of diseases. Dissection of
    miRNA-related associations are valuable for contemplating their mechanism in diseases,
    leading to the discovery of novel miRNAs for disease prognosis, diagnosis, and therapy.
    MOTIVATION: Apart from databases and prediction tools, miRNA-related information is largely
    available as unstructured text. Manual retrieval of these associations can be labor-intensive
    due to steadily growing number of publications. Additionally, most of the published miRNA
    entity recognition methods are keyword based, further subjected to manual inspection for
    retrieval of relations. Despite the fact that several databases host miRNA-associations
    derived from text, lower sensitivity and lack of published details for miRNA entity
    recognition and associated relations identification has motivated the need for developing
    comprehensive methods that are freely available for the scientific community. Additionally,
    the lack of a standard corpus for miRNA-relations has caused difficulty in evaluating the
    available systems. We propose methods to automatically extract mentions of miRNAs, species,
    genes/proteins, disease, and relations from scientific literature. Our generated corpora,
    along with dictionaries, and miRNA regular expression are freely available for academic
    purposes. To our knowledge, these resources are the most comprehensive developed so far.
    RESULTS: The identification of specific miRNA mentions reaches a recall of 0.94 and
    precision of 0.93. Extraction of miRNA-disease and miRNA-gene relations lead to an
    F1 score of up to 0.76. A comparison of the information extracted by our approach to
    the databases miR2Disease and miRSel for the extraction of Alzheimer's disease
    related relations shows the capability of our proposed methods in identifying correct
    relations with improved sensitivity. The published resources and described methods can
    help the researchers for maximal retrieval of miRNA-relations and generation of
    miRNA-regulatory networks. AVAILABILITY: The training and test corpora, annotation
    guidelines, developed dictionaries, and supplementary files are available at
    http://www.scai.fraunhofer.de/mirna-corpora.html.
},
note={26535109[pmid]},
note={PMC4602280[pmcid]},
issn={2046-1402},
url={https://pubmed.ncbi.nlm.nih.gov/26535109},
language={eng}
}
"""

_DATASETNAME = "mirna"
_DISPLAYNAME = "miRNA"

_DESCRIPTION = """\
The corpus consists of 301 Medline citations. The documents were screened for
mentions of miRNA in the abstract text. Gene, disease and miRNA entities were manually
annotated. The corpus comprises of two separate files, a train and a test set, coming
from 201 and 100 documents respectively. 
"""

_HOMEPAGE = "https://www.scai.fraunhofer.de/en/business-research-areas/bioinformatics/downloads/download-mirna-test-corpus.html"

_LICENSE = Licenses.CC_BY_NC_3p0

_BASE = "https://www.scai.fraunhofer.de/content/dam/scai/de/downloads/bioinformatik/miRNA/miRNA-"

_URLs = {
    "source": {
        "train": _BASE + "Train-Corpus.xml",
        "test": _BASE + "Test-Corpus.xml",
    },
    "bigbio_kb": {
        "train": _BASE + "Train-Corpus.xml",
        "test": _BASE + "Test-Corpus.xml",
    },
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.NAMED_ENTITY_DISAMBIGUATION]
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class miRNADataset(datasets.GeneratorBasedBuilder):
    """mirna"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="mirna_source",
            version=SOURCE_VERSION,
            description="mirna source schema",
            schema="source",
            subset_id="mirna",
        ),
        BigBioConfig(
            name="mirna_bigbio_kb",
            version=BIGBIO_VERSION,
            description="mirna BigBio schema",
            schema="bigbio_kb",
            subset_id="mirna",
        ),
    ]

    DEFAULT_CONFIG_NAME = "mirna_source"

    def _info(self):

        if self.config.schema == "source":

            features = datasets.Features(
                {
                    "passages": [
                        {
                            "document_id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "text": datasets.Value("string"),
                            "offset": datasets.Value("int32"),
                            "entities": [
                                {
                                    "id": datasets.Value("string"),
                                    "offsets": [[datasets.Value("int32")]],
                                    "text": [datasets.Value("string")],
                                    "type": datasets.Value("string"),
                                    "normalized": [
                                        {
                                            "db_name": datasets.Value("string"),
                                            "db_id": datasets.Value("string"),
                                        }
                                    ],
                                }
                            ],
                        }
                    ]
                }
            )

        elif self.config.schema == "bigbio_kb":
            features = schemas.kb_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        my_urls = _URLs[self.config.schema]

        path_xml_train = dl_manager.download(my_urls["train"])
        path_xml_test = dl_manager.download(my_urls["test"])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": path_xml_train,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": path_xml_test,
                    "split": "test",
                },
            ),
        ]

    def _get_passages_and_entities(self, d) -> Tuple[List[Dict], List[List[Dict]]]:

        sentences: List[Dict] = []
        entities: List[List[Dict]] = []
        relations: List[List[Dict]] = []

        text_total_length = 0

        po_start = 0

        # Get sentences of the document
        for _, s in enumerate(d):

            # annotation used only for document indexing
            if s.attrib["text"] is None or len(s.attrib["text"]) <= 0:
                continue

            # annotation used only for document indexing
            if len(s) <= 0:
                continue

            text_total_length += len(s.attrib["text"]) + 1

            po_end = po_start + len(s.attrib["text"])

            start = po_start

            dp = {
                "text": s.attrib["text"],
                "type": "title" if ".s0" in s.attrib["id"] else "abstract",
                "offsets": [(po_start, po_end)],
                "offset": 0,  # original offset
            }

            po_start = po_end + 1

            sentences.append(dp)

            pe = []  # entities
            re = []  # relations

            # For each entity
            for a in s:

                # If correspond to a entity
                if a.tag == "entity":

                    length = len(a.attrib["text"])

                    if a.attrib["text"] is None or length <= 0:
                        continue

                    # no in-text annotation: only for document indexing
                    if a.attrib["type"] in ["MeSH_Indexing_Chemical", "OTHER"]:
                        continue

                    startOffset, endOffset = a.attrib["charOffset"].split("-")
                    startOffset, endOffset = int(startOffset), int(endOffset)

                    pe.append(
                        {
                            "id": a.attrib["id"],
                            "type": a.attrib["type"],
                            "text": (a.attrib["text"],),
                            "offsets": [(start + startOffset, start + endOffset + 1)],
                            "normalized": [
                                {"db_name": "miRNA-corpus", "db_id": a.attrib["id"]}
                            ],
                        }
                    )

                # If correspond to relation pair
                elif a.tag == "pair":

                    re.append(
                        {
                            "id": a.attrib["id"],
                            "type": a.attrib["type"],
                            "arg1_id": a.attrib["e1"],
                            "arg2_id": a.attrib["e2"],
                            "normalized": [],
                        }
                    )

            entities.append(pe)
            relations.append(re)

        return sentences, entities, relations

    def _generate_examples(
        self,
        filepath: str,
        split: str,
    ) -> Iterator[Tuple[int, Dict]]:
        """Yields examples as (key, example) tuples."""

        reader = ET.fromstring(open(str(filepath), "r").read())

        if self.config.schema == "source":

            for uid, doc in enumerate(reader):

                (
                    sentences,
                    sentences_entities,
                    relations,
                ) = self._get_passages_and_entities(doc)

                if (
                    len(sentences) < 1
                    or len(sentences_entities) < 1
                    or len(sentences_entities) != len(sentences)
                ):
                    continue

                for p, pe, re in zip(sentences, sentences_entities, relations):

                    p.pop("offsets")  # BioC has only start for passages offsets

                    p["document_id"] = doc.attrib["id"]
                    p["entities"] = pe  # BioC has per passage entities

                yield uid, {"passages": sentences}

        elif self.config.schema == "bigbio_kb":

            uid = 0

            for idx, doc in enumerate(reader):

                (
                    sentences,
                    sentences_entities,
                    relations,
                ) = self._get_passages_and_entities(doc)

                if (
                    len(sentences) < 1
                    or len(sentences_entities) < 1
                    or len(sentences_entities) != len(sentences)
                ):
                    continue

                # global id
                uid += 1

                # unpack per-sentence entities
                entities = [e for pe in sentences_entities for e in pe]

                for p in sentences:
                    p.pop("offset")  # drop original offset
                    p["text"] = (p["text"],)  # text in sentence is Sequence
                    p["id"] = uid
                    uid += 1

                for e in entities:
                    e["id"] = uid
                    uid += 1

                # unpack per-sentence relations
                relations = [r for re in relations for r in re]

                for r in relations:
                    r["id"] = uid
                    uid += 1

                yield idx, {
                    "id": uid,
                    "document_id": doc.attrib["id"],
                    "passages": sentences,
                    "entities": entities,
                    "events": [],
                    "coreferences": [],
                    "relations": relations,
                }
