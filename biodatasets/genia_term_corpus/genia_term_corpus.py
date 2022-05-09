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

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Tasks

_LOCAL = False
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

_LICENSE = """GENIA Project License for Annotated Corpora"""

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
    try:
        relative_offset = text.index(entity["text"], relative_offset)
    except ValueError:
        # Skip duplicated annotations:
        # <cons lex="tumour_cell" sem="G#cell_type"><cons lex="tumour_cell" sem="G#cell_type">tumour cells</cons></cons>
        return None, None
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
