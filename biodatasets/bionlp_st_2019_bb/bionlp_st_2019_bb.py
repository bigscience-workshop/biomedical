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
from typing import List

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
        "train": "https://drive.google.com/uc?id=1WMl9eD4OZXq8zkkmHp3hSEvAqaAVui6L/view",
        "dev": "https://drive.google.com/uc?id=1oOfOfjIfg1XnesXwaKvSDfqgnchuximG/view",
        "test": "https://drive.google.com/uc?id=1_dRbgpGJUBCfF-iN2qOAgOBRvYmE7byW/view",
    },
    "bigbio_kb": {
        "train": "https://drive.google.com/uc?id=1WMl9eD4OZXq8zkkmHp3hSEvAqaAVui6L/view",
        "dev": "https://drive.google.com/uc?id=1oOfOfjIfg1XnesXwaKvSDfqgnchuximG/view",
        "test": "https://drive.google.com/uc?id=1_dRbgpGJUBCfF-iN2qOAgOBRvYmE7byW/view",
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
            name="bionlp_st_2019_bb_source",
            version=SOURCE_VERSION,
            description="bionlp_st_2019_bb source schema",
            schema="source",
            subset_id="bionlp_st_2019_bb",
        ),
        BigBioConfig(
            name="bionlp_st_2019_bb_bigbio_kb",
            version=BIGBIO_VERSION,
            description="bionlp_st_2019_bb BigBio schema",
            schema="bigbio_kb",
            subset_id="bionlp_st_2019_bb",
        ),
    ]

    DEFAULT_CONFIG_NAME = "bionlp_st_2019_bb_source"

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
                            "type": datasets.Value("string"),
                            "ref_id": datasets.Value("string"),
                            "resource_name": datasets.Value("string"),  # Name of the resource, e.g. "Wikipedia"
                            "cuid": datasets.Value("string"),  # ID in the resource, e.g. 534366
                            "text": datasets.Value(
                                "string"
                            ),  # Human readable description/name of the entity, e.g. "Barack Obama"
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

        my_urls = _URLs[self.config.schema]
        #data_dir = Path(dl_manager.download_and_extract(my_urls))
        data_files = {
            "train": Path(dl_manager.download_and_extract(my_urls["train"])) / f"",
            "dev": Path(dl_manager.download_and_extract(my_urls["dev"])) / f"",
            "test": Path(dl_manager.download_and_extract(my_urls["test"])) / f"",
        }
        breakpoint()
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
                example = parsing.parse_brat_file(txt_file)
                example["id"] = str(guid)
                yield guid, example
        elif self.config.schema == "bigbio_kb":
            txt_files = list(data_files.glob("*txt"))
            for guid, txt_file in enumerate(txt_files):
                example = parsing.brat_parse_to_bigbio_kb(
                    parsing.parse_brat_file(txt_file), entity_types=self._ENTITY_TYPES
                )
                example["id"] = str(guid)
                yield guid, example
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
