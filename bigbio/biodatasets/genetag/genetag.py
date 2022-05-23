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
Named entity recognition (NER) is an important first step for text mining the biomedical literature.
Evaluating the performance of biomedical NER systems is impossible without a standardized test corpus.
The annotation of such a corpus for gene/protein name NER is a difficult process due to the complexity
of gene/protein names. We describe the construction and annotation of GENETAG, a corpus of 20K MEDLINE®
sentences for gene/protein NER. 15K GENETAG sentences were used for the BioCreAtIvE Task 1A Competition.
"""


import re
from typing import Dict, List, Tuple

import datasets

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Tasks

_LANGUAGES = [Lang.EN]
_LOCAL = False
_CITATION = """\
@article{Tanabe2005,
  author    = {Lorraine Tanabe and Natalie Xie and Lynne H Thom and Wayne Matten and W John Wilbur},
  title     = {{GENETAG}: a tagged corpus for gene/protein named entity recognition},
  journal   = {{BMC} Bioinformatics},
  volume    = {6},
  year      = {2005},
  url       = {https://doi.org/10.1186/1471-2105-6-S1-S3},
  doi       = {10.1186/1471-2105-6-s1-s3},
  biburl    = {},
  bibsource = {}
}
"""

_DATASETNAME = "genetag"

_DESCRIPTION = """\
Named entity recognition (NER) is an important first step for text mining the biomedical literature.
Evaluating the performance of biomedical NER systems is impossible without a standardized test corpus.
The annotation of such a corpus for gene/protein name NER is a difficult process due to the complexity
of gene/protein names. We describe the construction and annotation of GENETAG, a corpus of 20K MEDLINE®
sentences for gene/protein NER. 15K GENETAG sentences were used for the BioCreAtIvE Task 1A Competition..
"""

_HOMEPAGE = "https://github.com/openbiocorpora/genetag"

_LICENSE = "Public Domain"

_BASE_URL = "https://raw.githubusercontent.com/openbiocorpora/genetag/master/original-data/"

_URLS = {
    "test": {
        "correct": f"{_BASE_URL}test/Correct.Data",
        "gold": f"{_BASE_URL}test/Gold.format",
        "text": f"{_BASE_URL}test/TOKENIZED_CORPUS",
        "postagspath": f"{_BASE_URL}test/TAGGED_GENE_CORPUS",
    },
    "train": {
        "correct": f"{_BASE_URL}train/Correct.Data",
        "gold": f"{_BASE_URL}train/Gold.format",
        "text": f"{_BASE_URL}train/TOKENIZED_CORPUS",
        "postagspath": f"{_BASE_URL}train/TAGGED_GENE_CORPUS",
    },
    "round1": {
        "correct": f"{_BASE_URL}round1/Correct.Data",
        "gold": f"{_BASE_URL}round1/Gold.format",
        "text": f"{_BASE_URL}round1/TOKENIZED_CORPUS",
        "postagspath": f"{_BASE_URL}round1/TAGGED_GENE_CORPUS",
    },
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class GenetagDataset(datasets.GeneratorBasedBuilder):
    """GENETAG is a corpus of 15K MEDLINE sentences with annotations for gene/protein NER"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = []
    for annot_type in ["gold", "correct"]:
        BUILDER_CONFIGS.append(
            BigBioConfig(
                name=f"genetag{annot_type}_source",
                version=SOURCE_VERSION,
                description=f"GENETAG {annot_type} annotation source schema",
                schema="source",
                subset_id=f"genetag{annot_type}",
            )
        )

        BUILDER_CONFIGS.append(
            BigBioConfig(
                name=f"genetag{annot_type}_bigbio_kb",
                version=BIGBIO_VERSION,
                description=f"GENETAG {annot_type} annotation bigbio schema",
                schema="bigbio_kb",
                subset_id=f"genetag{annot_type}",
            )
        )

    DEFAULT_CONFIG_NAME = "genetaggold_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "doc_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "tokenized_text": datasets.Sequence(datasets.Value("string")),
                    "pos_tags": datasets.Sequence(datasets.Value("string")),
                    "entities": [
                        {
                            "token_offsets": datasets.Sequence([datasets.Value("int32")]),
                            "text": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "entity_id": datasets.Value("string"),
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
        urls = _URLS
        data_dir = dl_manager.download_and_extract(urls)
        annotation_type = self.config.subset_id.split("genetag")[-1]  # correct or gold annotations

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir["train"]["text"],
                    "annotationpath": data_dir["train"][annotation_type],
                    "postagspath": data_dir["train"]["postagspath"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir["test"]["text"],
                    "annotationpath": data_dir["test"][annotation_type],
                    "postagspath": data_dir["test"]["postagspath"],
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir["round1"]["text"],
                    "annotationpath": data_dir["round1"][annotation_type],
                    "postagspath": data_dir["round1"]["postagspath"],
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath, annotationpath, postagspath, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        corpus, annotations = self._read_files(filepath, annotationpath, postagspath)

        if self.config.schema == "source":
            source_examples = self._parse_annotations_source(corpus, annotations, split)
            for uid, doc_id in enumerate(source_examples):
                yield uid, source_examples[doc_id]

        elif self.config.schema == "bigbio_kb":
            bb_kb_examples = self._parse_annotations_bb(corpus, annotations, split)
            for uid, doc_id in enumerate(bb_kb_examples):
                yield uid, bb_kb_examples[doc_id]

    def _read_files(self, filepath, annotation_path, postagspath):
        """
        Reads text corpus and annotations
        """
        corpus, annotations = dict(), dict()

        # read corpus
        with open(filepath, "r") as texts:
            for line in texts:
                # "@@95229799480" from "@@95229799480 Cervicovaginal ..."
                sentence_id = re.search(r"@@\d+", line).group(0)
                # remove "/TAG" suffix and "./"
                text = re.sub(r"(/TAG|\/\.)", "", line).split(sentence_id)[-1].strip()
                corpus[sentence_id] = {
                    "text": text,
                    "tokenized_text": text.split(),  # every token is space separated at source
                }

        with open(postagspath, "r") as texts:
            for line in texts:
                sentence_id = re.search(r"@@\d+", line).group(0)
                _tags = re.findall(r"(\/[A-Z]+|\/[.,:()\"]+)", line)
                pos_tags = [i.replace("/", "") for i in _tags]
                corpus[sentence_id]["pos_tags"] = pos_tags

        # read annotations
        with open(annotation_path, "r") as annots:
            for line in annots:
                row = line.split("|")
                if len(row) == 3:
                    sentence_id = row[0].strip()
                    annot = row[2].strip()
                    start = int(row[1].split()[0])
                    end = int(row[1].split()[1])
                    if sentence_id in annotations:
                        annotations[sentence_id].append({"text": annot, "token_start": start, "token_end": end})
                    else:
                        annotations[sentence_id] = [{"text": annot, "token_start": start, "token_end": end}]

        return corpus, annotations

    def _parse_annotations_source(self, corpus, annotations, split) -> Dict:
        """
        Reads source annotations
        """
        # Convert to source schema
        source_examples = {}
        for sent_id in corpus:

            text = corpus[sent_id]["text"]
            source_examples[sent_id] = {
                "doc_id": sent_id,
                "text": text,
                "tokenized_text": corpus[sent_id]["tokenized_text"],
                "pos_tags": corpus[sent_id]["pos_tags"],
                "entities": [],
            }

            if annotations.get(sent_id):
                for uid, entity in enumerate(annotations[sent_id]):
                    source_examples[sent_id]["entities"].append(
                        {
                            "text": entity["text"],
                            "type": "NEWGENE",
                            "token_offsets": [[entity["token_start"], entity["token_end"]]],
                            "entity_id": f"{sent_id}_{uid+1}",
                        }
                    )

        return source_examples

    def _parse_annotations_bb(self, corpus, annotations, split) -> Dict:
        """
        Convert source annotations to bigbio schema annotations
        """
        bb_examples = {}

        for sent_id in corpus:
            text = corpus[sent_id]["text"]
            bb_examples[sent_id] = {
                "id": sent_id,
                "document_id": sent_id,
                "passages": [
                    {"id": f"{sent_id}_text", "type": "sentence", "text": [text], "offsets": [[0, len(text)]]}
                ],
                "entities": self._add_entities_bb(sent_id, annotations[sent_id], text)
                if annotations.get(sent_id)
                else [],
                "events": [],
                "coreferences": [],
                "relations": [],
            }

        return bb_examples

    def _add_entities_bb(self, doc_id, annotations, text) -> List:
        """
        Returns entities in bigbio schema when given annotations
        (with token indices) for some text
        a text. e.g: -

        doc_id: @@21234669976
        annotations: [{'text': 'HLH', 'token_start': 9, 'token_end': 9},
                      {'text': 'AP-4 HLH', 'token_start': 8, 'token_end': 9},
                      {'text': 'AP-4 HLH motif', 'token_start': 8, 'token_end': 10}]
        text: 'Like other members of this family , the AP-4 HLH motif and the adjacent
               basic domain are necessary and sufficient to confer site-specific DNA binding .'

        returns:  [
                    {'offsets': [[45, 48]],
                    'text': ['HLH'],
                    'type': 'NEWGENE',
                    'normalized': [],
                    'id': '@@21234669976_1'},
                    {'offsets': [[40, 48]],
                    'text': ['AP-4 HLH'],
                    'type': 'NEWGENE',
                    'normalized': [],
                    'id': '@@21234669976_2'},
                    {'offsets': [[40, 54]],
                    'text': ['AP-4 HLH motif'],
                    'type': 'NEWGENE',
                    'normalized': [],
                    'id': '@@21234669976_3'}
                ]

        Uses the given token level indices to pick correct entities
        and assign character offsets
        """

        entities = []
        for uid, entity in enumerate(annotations):
            start = entity["token_start"]
            end = entity["token_end"]
            for i in range(len(text)):

                if text[i:].startswith(entity["text"]):
                    # match substring using character and word index
                    token_end = end + 1
                    token_end_char = i + len(entity["text"])
                    if " ".join(text.split()[start:token_end]) == text[i:token_end_char]:
                        annot = {
                            "offsets": [[i, i + len(entity["text"])]],
                            "text": [entity["text"]],
                            "type": "NEWGENE",
                            "normalized": [],
                        }
                        if annot not in entities:
                            annot["id"] = f"{doc_id}_{uid+1}"
                            entities.append(annot)
                            break
        return entities
