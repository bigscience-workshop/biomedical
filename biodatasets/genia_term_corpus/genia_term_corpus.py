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
The identification of linguistic expressions referring to entities of interest in molecular biology such as proteins,
genes and cells is a fundamental task in biomolecular text mining. The GENIA technical term annotation covers the
identification of  physical biological entities as well as other important terms. The corpus annotation covers the full
1,999 abstracts of the primary GENIA corpus.
"""

import xml.etree.ElementTree as ET
from itertools import count
from typing import Dict, List, Tuple

import datasets

from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

_CITATION = """\
@inproceedings{10.5555/1289189.1289260,
author = {Ohta, Tomoko and Tateisi, Yuka and Kim, Jin-Dong},
title = {The GENIA Corpus: An Annotated Research Abstract Corpus in Molecular Biology Domain},
year = {2002},
publisher = {Morgan Kaufmann Publishers Inc.},
address = {San Francisco, CA, USA},
booktitle = {Proceedings of the Second International Conference on Human Language Technology Research},
pages = {82–86},
numpages = {5},
location = {San Diego, California},
series = {HLT '02}
}

@article{Kim2003GENIAC,
  title={GENIA corpus - a semantically annotated corpus for bio-textmining},
  author={Jin-Dong Kim and Tomoko Ohta and Yuka Tateisi and Junichi Tsujii},
  journal={Bioinformatics},
  year={2003},
  volume={19 Suppl 1},
  pages={
          i180-2
        }
}

@inproceedings{10.5555/1567594.1567610,
author = {Kim, Jin-Dong and Ohta, Tomoko and Tsuruoka, Yoshimasa and Tateisi, Yuka and Collier, Nigel},
title = {Introduction to the Bio-Entity Recognition Task at JNLPBA},
year = {2004},
publisher = {Association for Computational Linguistics},
address = {USA},
booktitle = {Proceedings of the International Joint Workshop on Natural Language Processing in Biomedicine and Its
Applications},
pages = {70–75},
numpages = {6},
location = {Geneva, Switzerland},
series = {JNLPBA '04}
}
"""

_DATASETNAME = "genia_term_corpus"

_DESCRIPTION = """\
The identification of linguistic expressions referring to entities of interest in molecular biology such as proteins,
genes and cells is a fundamental task in biomolecular text mining. The GENIA technical term annotation covers the
identification of  physical biological entities as well as other important terms. The corpus annotation covers the full
1,999 abstracts of the primary GENIA corpus.
"""

_HOMEPAGE = "http://www.geniaproject.org/genia-corpus/term-corpus"

_LICENSE = """GENIA Project License for Annotated Corpora

1. Copyright of abstracts

Any abstracts contained in this corpus are from PubMed(R), a database
of the U.S. National Library of Medicine (NLM).

NLM data are produced by a U.S. Government agency and include works of
the United States Government that are not protected by U.S. copyright
law but may be protected by non-US copyright law, as well as abstracts
originating from publications that may be protected by U.S. copyright
law.

NLM assumes no responsibility or liability associated with use of
copyrighted material, including transmitting, reproducing,
redistributing, or making commercial use of the data. NLM does not
provide legal advice regarding copyright, fair use, or other aspects
of intellectual property rights. Persons contemplating any type of
transmission or reproduction of copyrighted material such as abstracts
are advised to consult legal counsel.

2. Copyright of full texts

Any full texts contained in this corpus are from the PMC Open Access
Subset of PubMed Central (PMC), the U.S. National Institutes of Health
(NIH) free digital archive of biomedical and life sciences journal
literature.

Articles in the PMC Open Access Subset are protected by copyright, but
are made available under a Creative Commons or similar license that
generally allows more liberal redistribution and reuse than a
traditional copyrighted work. Please refer to the license of each
article for specific license terms.

3. Copyright of annotations

The copyrights of annotations created in the GENIA Project of Tsujii
Laboratory, University of Tokyo, belong in their entirety to the GENIA
Project.

4. Licence terms

Use and distribution of abstracts drawn from PubMed is subject to the
PubMed(R) license terms as stated in Clause 1.

Use and distribution of full texts is subject to the license terms
applying to each publication.

Annotations created by the GENIA Project are licensed under the
Creative Commons Attribution 3.0 Unported License. To view a copy of
this license, visit http://creativecommons.org/licenses/by/3.0/ or
send a letter to Creative Commons, 444 Castro Street, Suite 900,
Mountain View, California, 94041, USA.

Annotations created by the GENIA Project must be attributed as
detailed in Clause 5.

5. Attribution

The GENIA Project was founded and led by prof. Jun'ichi Tsujii and
the project and its annotation efforts have been coordinated in part
by Nigel Collier, Yuka Tateisi, Sang-Zoo Lee, Tomoko Ohta, Jin-Dong
Kim, and Sampo Pyysalo.

For a complete list of the GENIA Project members and contributors,
please refer to http://www.geniaproject.org.

The GENIA Project has been supported by Grant-in-Aid for Scientific
Research on Priority Area "Genome Information Science" (MEXT, Japan),
Grant-in-Aid for Scientific Research on Priority Area "Systems
Genomics" (MEXT, Japan), Core Research for Evolutional Science &
Technology (CREST) "Information Mobility Project" (JST, Japan),
Solution Oriented Research for Science and Technology (SORST) (JST,
Japan), Genome Network Project (MEXT, Japan) and Grant-in-Aid for
Specially Promoted Research (MEXT, Japan).

Annotations covered by this license must be attributed as follows:

    Corpus annotations (c) GENIA Project

Distributions including annotations covered by this licence must
include this license text and Attribution section.

6. References

- GENIA Project : http://www.geniaproject.org
- PubMed : http://www.pubmed.gov/
- NLM (United States National Library of Medicine) : http://www.nlm.nih.gov/
- MEXT (Ministry of Education, Culture, Sports, Science and Technology) : http://www.mext.go.jp/
- JST (Japan Science and Technology Agency) : http://www.jst.go.jp"""

_URLS = {
    _DATASETNAME: "http://www.nactem.ac.uk/GENIA/current/GENIA-corpus/Term/GENIAcorpus3.02.tgz",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "3.0.2"

_BIGBIO_VERSION = "1.0.0"


class GeniaTermCorpusDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="genia_term_corpus_source",
            version=SOURCE_VERSION,
            description="genia_term_corpus source schema",
            schema="source",
            subset_id="genia_term_corpus",
        ),
        BigBioConfig(
            name="genia_term_corpus_bigbio_kb",
            version=BIGBIO_VERSION,
            description="genia_term_corpus BigBio schema",
            schema="bigbio_kb",
            subset_id="genia_term_corpus",
        ),
    ]

    DEFAULT_CONFIG_NAME = "genia_term_corpus_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "document_id": datasets.Value("string"),
                    "title": [
                        {
                            "text": datasets.Value("string"),
                            "entities": [
                                {
                                    "text": datasets.Value("string"),
                                    "lex": datasets.Value("string"),
                                    "sem": datasets.Value("string"),
                                }
                            ],
                        }
                    ],
                    "abstract": [
                        {
                            "text": datasets.Value("string"),
                            "entities": [
                                {
                                    "text": datasets.Value("string"),
                                    "lex": datasets.Value("string"),
                                    "sem": datasets.Value("string"),
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
        data_dir = dl_manager.download(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "archive": dl_manager.iter_archive(data_dir),
                    "data_path": "GENIA_term_3.02/GENIAcorpus3.02.xml",
                },
            ),
        ]

    def _generate_examples(self, archive, data_path) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        uid = count(0)
        for path, file in archive:
            if path == data_path:
                for key, example in enumerate(iterparse_genia(file)):
                    if self.config.schema == "source":
                        yield key, example

                    elif self.config.schema == "bigbio_kb":
                        yield key, parse_genia_to_bigbio(example, uid)


def iterparse_genia(file):
    # ontology = None
    for _, element in ET.iterparse(file):
        # if element.tag == "import":
        #     ontology = {"name": element.get("resource"), "prefix": element.get("prefix")}
        if element.tag == "article":
            bibliomisc = element.find("articleinfo/bibliomisc").text
            document_id = parse_genia_bibliomisc(bibliomisc)
            title = element.find("title")
            title_sentences = parse_genia_sentences(title)
            abstract = element.find("abstract")
            abstract_sentences = parse_genia_sentences(abstract)
            yield {
                "document_id": document_id,
                "title": title_sentences,
                "abstract": abstract_sentences,
            }


def parse_genia_sentences(passage):
    sentences = []
    for sentence in passage.iter(tag="sentence"):
        text = "".join(sentence.itertext())
        entities = []
        for entity in sentence.iter(tag="cons"):  # constituent
            entity_lex = entity.get("lex", "")
            entity_sem = parse_genia_sem(entity.get("sem", ""))
            entity_text = "".join(entity.itertext())
            entities.append({"text": entity_text, "lex": entity_lex, "sem": entity_sem})
        sentences.append(
            {
                "text": text,
                "entities": entities,
            }
        )
    return sentences


def parse_genia_bibliomisc(bibliomisc):
    """Remove 'MEDLINE:' from 'MEDLINE:96055286'."""
    return bibliomisc.replace("MEDLINE:", "") if ":" in bibliomisc else bibliomisc


def parse_genia_sem(sem):
    return sem.replace("G#", "") if "G#" in sem else sem


def parse_genia_to_bigbio(example, uid):
    document = {
        "id": next(uid),
        "document_id": example["document_id"],
        "passages": list(generate_bigbio_passages(example, uid)),
        "entities": list(generate_bigbio_entities(example, uid)),
        "events": [],
        "coreferences": [],
        "relations": [],
    }
    return document


def parse_genia_to_bigbio_passage(passage, uid, type="", offset=0):
    text = " ".join(sentence["text"] for sentence in passage)
    new_offset = offset + len(text)
    return {
        "id": next(uid),
        "type": type,
        "text": [text],
        "offsets": [[offset, new_offset]],
    }, new_offset + 1


def generate_bigbio_passages(example, uid):
    offset = 0
    for type in ["title", "abstract"]:
        passage, offset = parse_genia_to_bigbio_passage(example[type], uid, type=type, offset=offset)
        yield passage


def parse_genia_to_bigbio_entity(entity, uid, text="", relative_offset=0, offset=0):
    relative_offset = text.index(entity["text"], relative_offset)
    new_relative_offset = relative_offset + len(entity["text"])
    return {
        "id": next(uid),
        "offsets": [[offset + relative_offset, offset + new_relative_offset]],
        "text": [entity["text"]],
        "type": entity["sem"],
        "normalized": [],
    }, new_relative_offset


def generate_bigbio_entities(example, uid):
    sentence_offset = 0
    for type in ["title", "abstract"]:
        for sentence in example[type]:
            relative_offsets = {}
            for entity in sentence["entities"]:
                bigbio_entity, new_relative_offset = parse_genia_to_bigbio_entity(
                    entity,
                    uid,
                    text=sentence["text"],
                    relative_offset=relative_offsets.get((entity["text"], entity["lex"], entity["sem"]), 0),
                    offset=sentence_offset,
                )
                if bigbio_entity:
                    relative_offsets[(entity["text"], entity["lex"], entity["sem"])] = new_relative_offset
                    yield bigbio_entity
            sentence_offset += len(sentence["text"]) + 1
