# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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

from pathlib import Path
from typing import List, Dict

import datasets

from utils import parsing, schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

_DATASETNAME = "bionlp_st_2019_bb"
_SOURCE_VIEW_NAME = "source"
_UNIFIED_VIEW_NAME = "bigbio"

_CITATION = """\
@inproceedings{bossy-etal-2019-bacteria,
    title = "Bacteria Biotope at {B}io{NLP} Open Shared Tasks 2019",
    author = "Bossy, Robert  and
      Del{\'e}ger, Louise  and
      Chaix, Estelle  and
      Ba, Mouhamadou  and
      N{\'e}dellec, Claire",
    booktitle = "Proceedings of The 5th Workshop on BioNLP Open Shared Tasks",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D19-5719",
    doi = "10.18653/v1/D19-5719",
    pages = "121--131",
    abstract = "This paper presents the fourth edition of the Bacteria
    Biotope task at BioNLP Open Shared Tasks 2019. The task focuses on
    the extraction of the locations and phenotypes of microorganisms
    from PubMed abstracts and full-text excerpts, and the characterization
    of these entities with respect to reference knowledge sources (NCBI
    taxonomy, OntoBiotope ontology). The task is motivated by the importance
    of the knowledge on biodiversity for fundamental research and applications
    in microbiology. The paper describes the different proposed subtasks, the
    corpus characteristics, and the challenge organization. We also provide an
    analysis of the results obtained by participants, and inspect the evolution
    of the results since the last edition in 2016.",
    
}
"""

_DESCRIPTION = """\
The task focuses on the extraction of the locations and phenotypes of
microorganisms from PubMed abstracts and full-text excerpts, and the
characterization of these entities with respect to reference knowledge
sources (NCBI taxonomy, OntoBiotope ontology). The task is motivated by
the importance of the knowledge on biodiversity for fundamental research
and applications in microbiology.

"""

_HOMEPAGE = "https://sites.google.com/view/bb-2019/dataset"

_LICENSE = "https://creativecommons.org/licenses/by/3.0/ CC-BY-3.0"

_URLs = {
    "source": {
        "norm": {
            "train": "https://drive.google.com/uc?export=download&id=1aXbshxgytZ1Dhbmw7OULPFarPO1FbcM3",
            "dev": "https://drive.google.com/uc?export=download&id=14jRZWF8VeluEYrV9EybV3LeGm4q5nH6s",
            "test": "https://drive.google.com/uc?export=download&id=1BPDCFTVMmIlOowYA-DkeNNFjwTfHYPG6",
        },
        "norm+ner": {
            "train": "https://drive.google.com/uc?export=download&id=1yKxBPMej8EYdVeU4QS1xquFfXM76F-2K",
            "dev": "https://drive.google.com/uc?export=download&id=1Xk7h9bax533QWclO3Ur7aS07OATBF_bG",
            "test": "https://drive.google.com/uc?export=download&id=1Cb5hQIPS3LIeUL-UWdqyWfKB52xUz9cp",
        },
        "rel": {
            "train": "https://drive.google.com/uc?export=download&id=1gnc-ScNpssC3qrA7cVox4Iei7i96sYqC",
            "dev": "https://drive.google.com/uc?export=download&id=1wJM9XOfmvIBcX23t9bzQX5fLZwWQJIwS",
            "test": "https://drive.google.com/uc?export=download&id=1smhKA4LEPK5UJEyBLseq0mBaT9REUevu",
        },
        "rel+ner": {
            "train": "https://drive.google.com/uc?export=download&id=1CPx9NxTPQbygqMtxw3d0hNFajhecqgss",
            "dev": "https://drive.google.com/uc?export=download&id=1lVyCCuAJ5TmmTDz4S0dISBNiWGR745_7",
            "test": "https://drive.google.com/uc?export=download&id=1uE8oY5m-7mSA-W-e6vownnAVV97IwHhA",
        },
        "kb": {
            "train": "https://drive.google.com/uc?export=download&id=1Iuce3T_IArXWBbIJ7RXb_STaPnWKQBN-",
            "dev": "https://drive.google.com/uc?export=download&id=14yON_Tc9dm8esWYDVxL-krw23sgTCcdL",
            "test": "https://drive.google.com/uc?export=download&id=1wVqI_t9mirGUk71BkwkcKJv0VNGyaHDs",
        },
        "kb+ner": {
            "train": "https://drive.google.com/uc?export=download&id=1WMl9eD4OZXq8zkkmHp3hSEvAqaAVui6L",
            "dev": "https://drive.google.com/uc?export=download&id=1oOfOfjIfg1XnesXwaKvSDfqgnchuximG",
            "test": "https://drive.google.com/uc?export=download&id=1_dRbgpGJUBCfF-iN2qOAgOBRvYmE7byW",
        },
    },
    "bigbio": {
        "train": "https://drive.google.com/uc?export=download&id=1WMl9eD4OZXq8zkkmHp3hSEvAqaAVui6L",
        "dev": "https://drive.google.com/uc?export=download&id=1oOfOfjIfg1XnesXwaKvSDfqgnchuximG",
        "test": "https://drive.google.com/uc?export=download&id=1_dRbgpGJUBCfF-iN2qOAgOBRvYmE7byW",
    },
}

_SUPPORTED_TASKS = [
    Tasks.NAMED_ENTITY_RECOGNITION,
    Tasks.NAMED_ENTITY_DISAMBIGUATION,
    Tasks.RELATION_EXTRACTION,
]
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class bionlp_st_2019_bb(datasets.GeneratorBasedBuilder):
    """This dataset is the fourth edition of the Bacteria
    Biotope task at BioNLP Open Shared Tasks 2019"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="bionlp_st_2019_bb_norm_source",
            version=SOURCE_VERSION,
            description="bionlp_st_2019_bb entity normalization source schema",
            schema="source",
            subset_id="bionlp_st_2019_bb",
        ),
        BigBioConfig(
            name="bionlp_st_2019_bb_norm+ner_source",
            version=SOURCE_VERSION,
            description="bionlp_st_2019_bb entity recognition and normalization source schema",
            schema="source",
            subset_id="bionlp_st_2019_bb",
        ),
        BigBioConfig(
            name="bionlp_st_2019_bb_rel_source",
            version=SOURCE_VERSION,
            description="bionlp_st_2019_bb relation extraction source schema",
            schema="source",
            subset_id="bionlp_st_2019_bb",
        ),
        BigBioConfig(
            name="bionlp_st_2019_bb_rel+ner_source",
            version=SOURCE_VERSION,
            description="bionlp_st_2019_bb entity recognition and relation extraction source schema",
            schema="source",
            subset_id="bionlp_st_2019_bb",
        ),
        BigBioConfig(
            name="bionlp_st_2019_bb_kb_source",
            version=SOURCE_VERSION,
            description="bionlp_st_2019_bb entity normalization and relation extraction source schema",
            schema="source",
            subset_id="bionlp_st_2019_bb",
        ),
        BigBioConfig(
            name="bionlp_st_2019_bb_kb+ner_source",
            version=SOURCE_VERSION,
            description="bionlp_st_2019_bb entity recognition and normalization and relation extraction source schema",
            schema="source",
            subset_id="bionlp_st_2019_bb",
        ),
        #todo bigbio-kb
        BigBioConfig(
            name="bionlp_st_2019_bb_bigbio_kb",
            version=BIGBIO_VERSION,
            description="bionlp_st_2019_bb BigBio schema",
            schema="bigbio_kb",
            subset_id="bionlp_st_2019_bb",
        ),
    ]

    DEFAULT_CONFIG_NAME = "bionlp_st_2019_bb_kb+ner_source"

    _ENTITY_TYPES = {'Habitat',
                     'Microorganism',
                     'Phenotype',
                     'Geographical'}

    def _info(self):
        """
        - `features` defines the schema of the parsed data set. The schema depends on the
        chosen `config`: If it is `_SOURCE_VIEW_NAME` the schema is the schema of the
        original data. If `config` is `_UNIFIED_VIEW_NAME`, then the schema is the
        canonical KB-task schema defined in `biomedical/schemas/kb.py`.
        """
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "text_bound_annotations": [  # T line in brat, e.g. type or event trigger
                        {
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                            "text": datasets.Sequence(datasets.Value("string")),
                            "type": datasets.Value("string"),
                            "id": datasets.Value("string"),
                        }
                    ],
                    "events": [  # E line in brat
                        {
                            "trigger": datasets.Value("string"),  # refers to the text_bound_annotation of the trigger,
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "arguments": datasets.Sequence(
                                {
                                    "role": datasets.Value("string"),
                                    "ref_id": datasets.Value("string"),
                                }
                            ),
                        }
                    ],
                    "relations": [  # R line in brat
                        {
                            "id": datasets.Value("string"),
                            "head": {
                                "ref_id": datasets.Value("string"),
                                "role": datasets.Value("string"),
                            },
                            "tail": {
                                "ref_id": datasets.Value("string"),
                                "role": datasets.Value("string"),
                            },
                            "type": datasets.Value("string"),
                        }
                    ],
                    "equivalences": [  # Equiv line in brat
                        {
                            "id": datasets.Value("string"),
                            "ref_ids": datasets.Sequence(datasets.Value("string")),
                        }
                    ],
                    "attributes": [  # M or A lines in brat
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "ref_id": datasets.Value("string"),
                            "value": datasets.Value("string"),
                        }
                    ],
                    "normalizations": [  # N lines in brat
                        {
                            "id": datasets.Value("string"),
                            "ref_id": datasets.Value("string"),
                            "resource_name": datasets.Value("string"),  # Name of the resource, e.g. "Wikipedia"
                            "cuid": datasets.Value("string"),  # ID in the resource, e.g. 534366
                        }
                    ],
                },
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

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        version = self.config.name.split('_')[-2]
        my_urls = _URLs[self.config.schema][version]
        data_files = {
            "train": Path(dl_manager.download_and_extract(my_urls["train"])) / f"BioNLP-OST-2019_BB-{version}_train",
            "dev": Path(dl_manager.download_and_extract(my_urls["dev"])) / f"BioNLP-OST-2019_BB-{version}_dev",
            "test": Path(dl_manager.download_and_extract(my_urls["test"])) / f"BioNLP-OST-2019_BB-{version}_test",
        }
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_files": data_files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"data_files": data_files["dev"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"data_files": data_files["test"]},
            ),
        ]

    def _generate_examples(self, data_files: Path):
        if self.config.schema == "source":
            txt_files = list(data_files.glob("*txt"))
            for guid, txt_file in enumerate(txt_files):
                example = self.parse_brat_file(txt_file)
                example["id"] = str(guid)
                yield guid, example
        elif self.config.schema == "bigbio_kb":
            txt_files = list(data_files.glob("*txt"))
            for guid, txt_file in enumerate(txt_files):
                example = parsing.brat_parse_to_bigbio_kb(
                    self.parse_brat_file(txt_file), entity_types=self._ENTITY_TYPES
                )
                example["id"] = str(guid)
                yield guid, example
        else:
            raise ValueError(f"Invalid config: {self.config.name}")

    def parse_brat_file(self, txt_file: Path, annotation_file_suffixes: List[str] = None,
                        parse_notes: bool = False) -> Dict:
        """
        Parse a brat file into the schema defined below.
        `txt_file` should be the path to the brat '.txt' file you want to parse, e.g. 'data/1234.txt'
        Assumes that the annotations are contained in one or more of the corresponding '.a1', '.a2' or '.ann' files,
        e.g. 'data/1234.ann' or 'data/1234.a1' and 'data/1234.a2'.

        Will include annotator notes, when `parse_notes == True`.

        brat_features = datasets.Features(
            {
                "id": datasets.Value("string"),
                "document_id": datasets.Value("string"),
                "text": datasets.Value("string"),
                "text_bound_annotations": [  # T line in brat, e.g. type or event trigger
                    {
                        "offsets": datasets.Sequence([datasets.Value("int32")]),
                        "text": datasets.Sequence(datasets.Value("string")),
                        "type": datasets.Value("string"),
                        "id": datasets.Value("string"),
                    }
                ],
                "events": [  # E line in brat
                    {
                        "trigger": datasets.Value(
                            "string"
                        ),  # refers to the text_bound_annotation of the trigger,
                        "id": datasets.Value("string"),
                        "type": datasets.Value("string"),
                        "arguments": datasets.Sequence(
                            {
                                "role": datasets.Value("string"),
                                "ref_id": datasets.Value("string"),
                            }
                        ),
                    }
                ],
                "relations": [  # R line in brat
                    {
                        "id": datasets.Value("string"),
                        "head": {
                            "ref_id": datasets.Value("string"),
                            "role": datasets.Value("string"),
                        },
                        "tail": {
                            "ref_id": datasets.Value("string"),
                            "role": datasets.Value("string"),
                        },
                        "type": datasets.Value("string"),
                    }
                ],
                "equivalences": [  # Equiv line in brat
                    {
                        "id": datasets.Value("string"),
                        "ref_ids": datasets.Sequence(datasets.Value("string")),
                    }
                ],
                "attributes": [  # M or A lines in brat
                    {
                        "id": datasets.Value("string"),
                        "type": datasets.Value("string"),
                        "ref_id": datasets.Value("string"),
                        "value": datasets.Value("string"),
                    }
                ],
                "normalizations": [  # N lines in brat
                    {
                        "id": datasets.Value("string"),
                        "type": datasets.Value("string"),
                        "ref_id": datasets.Value("string"),
                        "resource_name": datasets.Value(
                            "string"
                        ),  # Name of the resource, e.g. "Wikipedia"
                        "cuid": datasets.Value(
                            "string"
                        ),  # ID in the resource, e.g. 534366
                        "text": datasets.Value(
                            "string"
                        ),  # Human readable description/name of the entity, e.g. "Barack Obama"
                    }
                ],
                ### OPTIONAL: Only included when `parse_notes == True`
                "notes": [  # # lines in brat
                    {
                        "id": datasets.Value("string"),
                        "type": datasets.Value("string"),
                        "ref_id": datasets.Value("string"),
                        "text": datasets.Value("string"),
                    }
                ],
            },
            )
        """

        example = {}
        example["document_id"] = txt_file.with_suffix("").name
        with txt_file.open(encoding='utf-8') as f:
            example["text"] = f.read()

        # If no specific suffixes of the to-be-read annotation files are given - take standard suffixes
        # for event extraction
        if annotation_file_suffixes is None:
            annotation_file_suffixes = [".a1", ".a2", ".ann"]

        if len(annotation_file_suffixes) == 0:
            raise AssertionError(
                "At least one suffix for the to-be-read annotation files should be given!"
            )

        ann_lines = []
        for suffix in annotation_file_suffixes:
            annotation_file = txt_file.with_suffix(suffix)
            if annotation_file.exists():
                with annotation_file.open(encoding='utf-8') as f:
                    ann_lines.extend(f.readlines())

        example["text_bound_annotations"] = []
        example["events"] = []
        example["relations"] = []
        example["equivalences"] = []
        example["attributes"] = []
        example["normalizations"] = []

        if parse_notes:
            example["notes"] = []

        for line in ann_lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("T"):  # Text bound
                ann = {}
                fields = line.split("\t")
                ann["id"] = fields[0]
                ann["type"] = fields[1].split()[0]
                if ann["type"] in ['Title', 'Paragraph']:
                    break
                ann["offsets"] = []
                span_str = parsing.remove_prefix(fields[1], (ann["type"] + " "))
                text = fields[2]
                for span in span_str.split(";"):
                    start, end = span.split()
                    ann["offsets"].append([int(start), int(end)])

                # Heuristically split text of discontiguous entities into chunks
                ann["text"] = []
                if len(ann["offsets"]) > 1:
                    i = 0
                    for start, end in ann["offsets"]:
                        chunk_len = end - start
                        ann["text"].append(text[i:chunk_len + i])
                        i += chunk_len
                        while i < len(text) and text[i] == " ":
                            i += 1
                else:
                    ann["text"] = [text]

                example["text_bound_annotations"].append(ann)

            elif line.startswith("E"):
                ann = {}
                fields = line.split("\t")

                ann["id"] = fields[0]

                ann["type"], ann["trigger"] = fields[1].split()[0].split(":")

                ann["arguments"] = []
                for role_ref_id in fields[1].split()[1:]:
                    argument = {
                        "role": (role_ref_id.split(":"))[0],
                        "ref_id": (role_ref_id.split(":"))[1],
                    }
                    ann["arguments"].append(argument)

                example["events"].append(ann)

            elif line.startswith("R"):
                ann = {}
                fields = line.split("\t")

                ann["id"] = fields[0]
                ann["type"] = fields[1].split()[0]

                ann["head"] = {
                    "role": fields[1].split()[1].split(":")[0],
                    "ref_id": fields[1].split()[1].split(":")[1],
                }
                ann["tail"] = {
                    "role": fields[1].split()[2].split(":")[0],
                    "ref_id": fields[1].split()[2].split(":")[1],
                }

                example["relations"].append(ann)

            # '*' seems to be the legacy way to mark equivalences,
            # but I couldn't find any info on the current way
            # this might have to be adapted dependent on the brat version
            # of the annotation
            elif line.startswith("*"):
                ann = {}
                fields = line.split("\t")

                ann["id"] = fields[0]
                ann["ref_ids"] = fields[1].split()[1:]

                example["equivalences"].append(ann)

            elif line.startswith("A") or line.startswith("M"):
                ann = {}
                fields = line.split("\t")

                ann["id"] = fields[0]

                info = fields[1].split()
                ann["type"] = info[0]
                ann["ref_id"] = info[1]

                if len(info) > 2:
                    ann["value"] = info[2]
                else:
                    ann["value"] = ""

                example["attributes"].append(ann)

            elif line.startswith("N"):
                ann = {}
                fields = line.split("\t")

                ann["id"] = fields[0]

                info = fields[1].split()

                ann["ref_id"] = info[1].split(':')[-1]
                ann["resource_name"] = info[0]
                ann["cuid"] = ''.join(info[2].split(":")[1:])
                example["normalizations"].append(ann)

            elif parse_notes and line.startswith("#"):
                ann = {}
                fields = line.split("\t")

                ann["id"] = fields[0]
                ann["text"] = fields[2]

                info = fields[1].split()

                ann["type"] = info[0]
                ann["ref_id"] = info[1]
                example["notes"].append(ann)

        return example