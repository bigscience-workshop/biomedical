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

import itertools as it
from typing import Dict, Generator, List, Tuple

import datasets

from .bigbiohub import BigBioConfig, Tasks, kb_features

_LANGUAGES = ["English"]
_PUBMED = True
_LOCAL = False

_CITATION = """\
@article{cho2022plant,
  author    = {Cho, Hyejin and Kim, Baeksoo and Choi, Wonjun and Lee, Doheon and Lee, Hyunju},
  title     = {Plant phenotype relationship corpus for biomedical relationships between plants and phenotypes},
  journal   = {Scientific Data},
  volume    = {9},
  year      = {2022},
  publisher = {Nature Publishing Group},
  doi       = {https://doi.org/10.1038/s41597-022-01350-1},
}
"""

_DATASETNAME = "ppr"
_DISPLAYNAME = "Plant-Phenotype-Relations"

_DESCRIPTION = """\
The Plant-Phenotype corpus is a text corpus with human annotations of plants, phenotypes, and their relations on a \
corpus in 600 PubMed abstracts.
"""

_HOMEPAGE = "https://github.com/DMCB-GIST/PPRcorpus"

_LICENSE = "UNKNOWN"

_URLS = {
    _DATASETNAME: [
        "https://raw.githubusercontent.com/davidkartchner/PPRcorpus/main/corpus/PPR_train_corpus.txt",
        "https://raw.githubusercontent.com/davidkartchner/PPRcorpus/main/corpus/PPR_dev_corpus.txt",
        "https://raw.githubusercontent.com/davidkartchner/PPRcorpus/main/corpus/PPR_test_corpus.txt",
    ],
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class PlantPhenotypeDataset(datasets.GeneratorBasedBuilder):
    """Plant-Phenotype is dataset for NER and RE of plants and their induced phenotypes"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="ppr_source",
            version=SOURCE_VERSION,
            description="Plant Phenotype Relations source schema",
            schema="source",
            subset_id="plant_phenotype",
        ),
        BigBioConfig(
            name="ppr_bigbio_kb",
            version=BIGBIO_VERSION,
            description="Plant Phenotype Relations BigBio schema",
            schema="bigbio_kb",
            subset_id="plant_phenotype",
        ),
    ]

    DEFAULT_CONFIG_NAME = "ppr_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":

            features = datasets.Features(
                {
                    "passage_id": datasets.Value("string"),
                    "pmid": datasets.Value("string"),
                    "section": datasets.Value("int32"),
                    "text": datasets.Value("string"),
                    "entities": [
                        {
                            "offsets": datasets.Sequence(datasets.Value("int32")),
                            "text": datasets.Value("string"),
                            "type": datasets.Value("string"),
                        }
                    ],
                    "relations": [
                        {
                            "relation_type": datasets.Value("string"),
                            "entity1_offsets": datasets.Sequence(datasets.Value("int32")),
                            "entity1_text": datasets.Value("string"),
                            "entity1_type": datasets.Value("string"),
                            "entity2_offsets": datasets.Sequence(datasets.Value("int32")),
                            "entity2_text": datasets.Value("string"),
                            "entity2_type": datasets.Value("string"),
                        }
                    ],
                }
            )

        elif self.config.schema == "bigbio_kb":
            features = kb_features
        else:
            raise NotImplementedError(f"Schema {self.config.schema} not supported")

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
        train, dev, test = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": train,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": test,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": dev,
                },
            ),
        ]

    def _generate_examples(
        self,
        filepath,
    ) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        with open(filepath, "r") as f:
            chunks = f.read().strip().split("\n\n")

        if self.config.schema == "source":
            for id_, doc in self._generate_source_examples(chunks):
                yield id_, doc

        elif self.config.schema == "bigbio_kb":
            for id_, doc in self._generate_bigbio_kb_examples(chunks):
                yield id_, doc

    def _generate_whole_documents(self, annotation_chunks: List[str]) -> Generator[Dict, None, None]:
        """Aggregate individual sentence annotations into whole abstracts.

        Args:
            annotation_chunks (List[str]): List of annotation chunks, i.e., a sentence with its annotations.
                For example:
                10072339_4	OBJECTIVE: A patient with possible airborne facial dermatitis to potato is described.
                10072339	44	61	facial dermatitis	Negative_phenotype
                10072339	65	71	potato	Plant

        Returns:
              Generator producing a dictionary containing the pmid of an article and all document chunks.
        """
        prev_pmid = None
        pmid = ""
        doc_chunks = []

        for chunk in annotation_chunks:
            lines = chunk.split("\n")

            # The first line is the sentence (format: <pmid>_<num>\t<sentence-text>)
            passage_info, passage_text = lines[0].split("\t")

            # Then annotations in Pubtator format
            annotations = [line.split("\t") for line in lines[1:]]

            # Get info on passage
            pmid, section = passage_info.split("_")
            if prev_pmid is None:
                prev_pmid = pmid

            elif prev_pmid != pmid:
                yield {"pmid": prev_pmid, "doc_chunks": doc_chunks}

                # Reset everything for next PMID
                prev_pmid = pmid
                doc_chunks = []

            doc_chunks.append(
                {
                    "passage": passage_text,
                    "annotations": annotations,
                    "sentence_id": passage_info,
                }
            )

        # Take care of last document
        yield {"pmid": pmid, "doc_chunks": doc_chunks}

    def _generate_source_examples(self, annotation_chunks: List[str]) -> Generator[Tuple[str, Dict], None, None]:
        """Generate examples in format of source schema

        Args:
            annotation_chunks (List[str]): List of annotation chunks.

        Returns:
            Generator of instance tuples (<key>, <instance-dict>)
        """
        for chunk in annotation_chunks:
            lines = chunk.split("\n")

            passage_id, passage_text = lines[0].split("\t")
            annotations = [line.split("\t") for line in lines[1:]]

            # Get info on passage
            pmid, section = passage_id.split("_")
            section = int(section)

            # Grab entities and relations
            entities = []
            relations = []
            for annotation in annotations:
                if len(annotation) == 5:
                    # It's an entity annotation
                    entities.append(
                        {
                            "offsets": (int(annotation[1]), int(annotation[2])),
                            "text": annotation[3],
                            "type": annotation[4],
                        }
                    )

                elif len(annotation) == 10:
                    # Relation annotation
                    relations.append(
                        {
                            "relation_type": annotation[1],
                            "entity1_offsets": (int(annotation[2]), int(annotation[3])),
                            "entity1_text": annotation[4],
                            "entity1_type": annotation[5],
                            "entity2_offsets": (int(annotation[6]), int(annotation[7])),
                            "entity2_text": annotation[8],
                            "entity2_type": annotation[9],
                        }
                    )
                else:
                    # This is a special case that occurs for a single data point
                    relations.append(
                        {
                            "relation_type": annotation[1],
                            "entity1_offsets": (int(annotation[2]), int(annotation[3])),
                            "entity1_text": annotation[4],
                            "entity1_type": annotation[5],
                            "entity2_offsets": (int(annotation[8]), int(annotation[9])),
                            "entity2_text": annotation[10],
                            "entity2_type": annotation[11],
                        }
                    )

                # Consolidate into document
                document = {
                    "passage_id": passage_id,
                    "pmid": pmid,
                    "section": section,
                    "text": passage_text,
                    "entities": entities,
                    "relations": relations,
                }

            yield passage_id, document

    def _generate_bigbio_kb_examples(self, annotation_chunks: List[str]):
        """Generator for training examples in bigbio_kb schema format.

        Args:
            annotation_chunks (List[str]): List of annotation chunks.

        Returns:
            Generator of instance tuples (<key>, <instance-dict>)
        """
        uid = it.count(1)
        for document in self._generate_whole_documents(annotation_chunks):
            pmid = document["pmid"]
            offset_delta = 0
            id_ = str(next(uid))

            passages = []
            entities = []
            relations = []

            # Iterate through each section of the article
            for text_section in document["doc_chunks"]:
                # Extract passages
                passage = text_section["passage"]
                passages.append(
                    {
                        "id": str(next(uid)),
                        "text": [passage],
                        "type": "sentence",
                        "offsets": [(offset_delta, offset_delta + len(passage))],
                    }
                )

                # Extract entities
                entities_sublist = []
                for annotation in text_section["annotations"]:
                    if len(annotation) == 5:
                        entities_sublist.append(
                            {
                                "id": str(next(uid)),
                                "type": annotation[4],
                                "text": [annotation[3]],
                                "offsets": [(int(annotation[1]) + offset_delta, int(annotation[2]) + offset_delta)],
                                "normalized": [],
                            }
                        )

                # Create mapping of offsets to entity_id
                ent2id = {tuple(x["offsets"]): x["id"] for x in entities_sublist}
                entities.extend(entities_sublist)

                # Extract relations
                for annotation in text_section["annotations"]:
                    if len(annotation) == 10:
                        e1_offsets = [(int(annotation[2]) + offset_delta, int(annotation[3]) + offset_delta)]
                        e2_offsets = [(int(annotation[6]) + offset_delta, int(annotation[7]) + offset_delta)]
                        relations.append(
                            {
                                "id": str(next(uid)),
                                "type": annotation[1],
                                "arg1_id": ent2id[tuple(e1_offsets)],
                                "arg2_id": ent2id[tuple(e2_offsets)],
                                "normalized": [],
                            }
                        )

                    # Special case for a single annotation
                    elif len(annotation) > 10:
                        e1_offsets = [(int(annotation[2]) + offset_delta, int(annotation[3]) + offset_delta)]
                        e2_offsets = [(int(annotation[8]) + offset_delta, int(annotation[9]) + offset_delta)]
                        relations.append(
                            {
                                "id": str(next(uid)),
                                "type": annotation[1],
                                "arg1_id": ent2id[tuple(e1_offsets)],
                                "arg2_id": ent2id[tuple(e2_offsets)],
                                "normalized": [],
                            }
                        )

                offset_delta += len(passage) + 1

            doc = {
                "id": id_,
                "document_id": pmid,
                "passages": passages,
                "entities": entities,
                "relations": relations,
                "events": [],
                "coreferences": [],
            }

            yield id_, doc
