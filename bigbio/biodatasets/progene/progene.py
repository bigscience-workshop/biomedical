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

import itertools
import os
import pathlib
from typing import Dict, Iterable, Iterator, List, Tuple

import datasets

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tags, Tasks
from bigbio.utils.license import Licenses

_TAGS = [Tags.GENE]
_LANGUAGES = [Lang.EN]
_PUBMED = True
_LOCAL = False
_CITATION = """\
@inproceedings{faessler-etal-2020-progene,
    title = "{P}ro{G}ene - A Large-scale, High-Quality Protein-Gene Annotated Benchmark Corpus",
    author = "Faessler, Erik  and
      Modersohn, Luise  and
      Lohr, Christina  and
      Hahn, Udo",
    booktitle = "Proceedings of the 12th Language Resources and Evaluation Conference",
    month = may,
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2020.lrec-1.564",
    pages = "4585--4596",
    abstract = "Genes and proteins constitute the fundamental entities of molecular genetics. We here introduce ProGene (formerly called FSU-PRGE), a corpus that reflects our efforts to cope with this important class of named entities within the framework of a long-lasting large-scale annotation campaign at the Jena University Language {\&} Information Engineering (JULIE) Lab. We assembled the entire corpus from 11 subcorpora covering various biological domains to achieve an overall subdomain-independent corpus. It consists of 3,308 MEDLINE abstracts with over 36k sentences and more than 960k tokens annotated with nearly 60k named entity mentions. Two annotators strove for carefully assigning entity mentions to classes of genes/proteins as well as families/groups, complexes, variants and enumerations of those where genes and proteins are represented by a single class. The main purpose of the corpus is to provide a large body of consistent and reliable annotations for supervised training and evaluation of machine learning algorithms in this relevant domain. Furthermore, we provide an evaluation of two state-of-the-art baseline systems {---} BioBert and flair {---} on the ProGene corpus. We make the evaluation datasets and the trained models available to encourage comparable evaluations of new methods in the future.",
    language = "English",
    ISBN = "979-10-95546-34-4",
}
"""

_DATASETNAME = "PROGENE"

_DESCRIPTION = """\
The Protein/Gene corpus was developed at the JULIE Lab Jena under supervision of Prof. Udo Hahn.
The executing scientist was Dr. Joachim Wermter.
The main annotator was Dr. Rico Pusch who is an expert in biology.
The corpus was developed in the context of the StemNet project (http://www.stemnet.de/).
"""

_HOMEPAGE = "https://zenodo.org/record/3698568#.YlVHqdNBxeg"

_LICENSE = Licenses.CC_BY_4p0

# using custom url: original distribution includes trained models (>25GB) and original dataset license allow for redistribution
_URLS = "https://huggingface.co/datasets/bigscience-biomedical/progene/resolve/main/crossvalidation_data.zip"

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "1.1.0"

_BIGBIO_VERSION = "1.0.0"


class ProgeneDataset(datasets.GeneratorBasedBuilder):
    """ProgeneDataset"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="progene_source",
            version=SOURCE_VERSION,
            description="PROGENE source schema",
            schema="source",
            subset_id="progene",
        ),
        BigBioConfig(
            name="progene_bigbio_kb",
            version=BIGBIO_VERSION,
            description="PROGENE BigBio schema",
            schema="bigbio_kb",
            subset_id="progene",
        ),
    ]

    DEFAULT_CONFIG_NAME = "progene_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            # This follows something similar to CONLL dataset that is in the IOB Format as well
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "tags": datasets.Sequence(datasets.Value("string")),
                }
            )

        elif self.config.schema == "bigbio_kb":
            features = schemas.kb_features
        else:
            raise ValueError(
                "config schema is one of source or bigbio_kb for Progene Dataset"
            )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        urls = _URLS
        dl_dir = dl_manager.download_and_extract(urls)
        dataset_dir = os.path.join(dl_dir, "crossvalidation_data")
        dataset_dir = pathlib.Path(dataset_dir)
        splits = []
        for split_num in range(0, 10):
            for file in dataset_dir.joinpath(f"flairSplit{split_num}").iterdir():
                if file.name == "train.txt":
                    split_id = f"split_{split_num}_{datasets.Split.TRAIN}"
                elif file.name == "dev.txt":
                    split_id = f"split_{split_num}_{datasets.Split.VALIDATION}"
                else:
                    split_id = f"split_{split_num}_{datasets.Split.TEST}"

                splits.append(
                    datasets.SplitGenerator(
                        name=split_id,
                        gen_kwargs={"filepath": file, "split_id": split_id},
                    )
                )

        return splits

    def _generate_examples(self, filepath, split_id: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        with open(filepath, "r") as fp:
            guid = 0
            tokens = []
            ner_tags = []
            entity_ids = 0
            for line in fp:
                if line == "" or line == "\n":
                    if tokens:
                        entities = self.iob_tags_to_entities(tokens, ner_tags)
                        entity_dicts = []

                        for entity in entities:
                            entity_text = [str(entity[0])]
                            entity_offset = [entity[1]]
                            entity_dict = {
                                "id": f"{split_id}_{entity_ids}_entity",
                                "type": "progene_text",
                                "text": entity_text,
                                "offsets": entity_offset,
                                "normalized": [],
                            }
                            entity_ids += 1
                            entity_dicts.append(entity_dict)

                        if self.config.schema == "source":
                            yield f"{split_id}_{guid}", {
                                "id": f"{split_id}_{guid}",
                                "tokens": tokens,
                                "tags": ner_tags,
                            }
                        elif self.config.schema == "bigbio_kb":
                            yield f"{split_id}_{guid}", {
                                "id": f"{split_id}_{guid}",
                                "document_id": f"{split_id}_{guid}",
                                "passages": [
                                    {
                                        "id": f"{split_id}_{guid}_passage",
                                        "type": "progene_text",
                                        "text": [" ".join(tokens)],
                                        "offsets": [[0, len(" ".join(tokens))]],
                                    }
                                ],
                                "entities": entity_dicts,
                                "events": [],
                                "coreferences": [],
                                "relations": [],
                            }
                        guid += 1
                        tokens = []
                        ner_tags = []
                else:
                    text_tags = line.split("\t")
                    token = text_tags[0].strip()
                    ner_tag = text_tags[1].strip()
                    tokens.append(token)
                    ner_tags.append(ner_tag)

            # residual tokens and tags at the end of the file
            entities = self.iob_tags_to_entities(tokens, ner_tags)
            entity_dicts = []
            for entity in entities:
                entity_text = [str(entity[0])]
                entity_offset = [entity[1]]
                entity_dict = {
                    "id": f"{split_id}_{entity_ids}_entity",
                    "type": "progene_text",
                    "text": entity_text,
                    "offsets": entity_offset,
                    "normalized": [],
                }
                entity_ids += 1
                entity_dicts.append(entity_dict)

            if self.config.schema == "source":
                yield f"{split_id}_{guid}", {
                    "id": f"{split_id}_{guid}",
                    "tokens": tokens,
                    "tags": ner_tags,
                }
            elif self.config.schema == "bigbio_kb":
                yield f"{split_id}_{guid}", {
                    "id": f"{split_id}_{guid}",
                    "document_id": f"{split_id}_{guid}",
                    "passages": [
                        {
                            "id": f"{split_id}_{guid}_passage",
                            "type": "progene_text",
                            "text": [" ".join(tokens)],
                            "offsets": [[0, len(" ".join(tokens))]],
                        }
                    ],
                    "entities": entity_dicts,
                    "events": [],
                    "coreferences": [],
                    "relations": [],
                }

    def iob_to_biluo(self, tags: Iterable[str]) -> List[str]:
        """Converts IOB tags to BILUO tags. This is taken from spacy.training.iob_utils"""
        out: List[str] = []
        tags = list(tags)
        while tags:
            out.extend(self._consume_os(tags))
            out.extend(self._consume_ent(tags))
        return out

    def _consume_os(self, tags: List[str]) -> Iterator[str]:
        while tags and tags[0] == "O":
            yield tags.pop(0)

    def _consume_ent(self, tags: List[str]) -> List[str]:
        if not tags:
            return []
        tag = tags.pop(0)
        target_in = "I" + tag[1:]
        target_last = "L" + tag[1:]
        length = 1
        while tags and tags[0] in {target_in, target_last}:
            length += 1
            tags.pop(0)
        label = tag[2:]
        if length == 1:
            if len(label) == 0:
                raise ValueError("Error parsing iob")
            return ["U-" + label]
        else:
            start = "B-" + label
            end = "L-" + label
            middle = [f"I-{label}" for _ in range(1, length - 1)]
            return [start] + middle + [end]

    def tags_to_entities(self, tags: Iterable[str]) -> List[Tuple[str, int, int]]:
        """This has been taken from spacy.training.iob_utils
        Note that the end index returned by this function is inclusive.
        To use it for Span creation, increment the end by 1."""
        entities = []
        start = None
        for i, tag in enumerate(tags):
            if tag is None or tag.startswith("-"):
                # TODO: We shouldn't be getting these malformed inputs. Fix this.
                if start is not None:
                    start = None
                else:
                    entities.append(("", i, i))
            elif tag.startswith("O"):
                pass
            elif tag.startswith("I"):
                if start is None:
                    raise ValueError("Error converting tags to entities")
            elif tag.startswith("U"):
                entities.append((tag[2:], i, i))
            elif tag.startswith("B"):
                start = i
            elif tag.startswith("L"):
                if start is None:
                    raise ValueError("Error converting tags to entities")
                entities.append((tag[2:], start, i))
                start = None
            else:
                raise ValueError("Error converting tags to entities")
        return entities

    def iob_tags_to_entities(self, text: List[str], tags: List[str]):
        """Converts IOB Tags to a set of entities
        text: List[str] - A list of tokens
        tags: List[str] - A list of corresponding tags
        """

        assert len(text) == len(tags)

        biluo_tags = self.iob_to_biluo(tags)
        entity_offsets = self.tags_to_entities(biluo_tags)
        spans = self.get_span_offsets(" ".join(text))
        entities = []
        text_string = " ".join(text)
        for entity, start_word, end_word in entity_offsets:
            start_char = spans[start_word][0]
            end_char = (
                spans[end_word][1] - 1
            )  # The offsets include the space in the text
            entity_text = text_string[start_char:end_char]
            entity_offsets = [start_char, end_char]
            entities.append((entity_text, entity_offsets))

        return entities

    def get_span_offsets(self, text):
        """Returns the character offsets for every word in the text.
        We assume that every word ends in a space for this function
        """
        words = text.split()
        len_words = list(map(lambda word: len(word) + 1, words))
        offsets = [0] + len_words
        offsets = itertools.accumulate(offsets)
        offsets = list(offsets)
        offsets = list(zip(offsets, offsets[1:]))
        return offsets


if __name__ == "__main__":
    datasets.load_dataset(__file__, name="progene_bigbio_kb")
