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
Post-translational-modiﬁcations (PTM), amino acid modiﬁcations of proteins after translation, are one of the posterior
processes of protein biosynthesis for many proteins, and they are critical for determining protein function such as its
activity state, localization, turnover and interactions with other biomolecules. While there have been many studies of
information extraction targeting individual PTM types, there was until recently little effort to address extraction of
multiple PTM types at once in a unified framework.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict

import datasets
from bigbio.utils import parsing, schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks

_LANGUAGES = [Lang.EN]
_LOCAL = False
_CITATION = """\
@inproceedings{ohta-etal-2010-event,
    title = "Event Extraction for Post-Translational Modifications",
    author = "Ohta, Tomoko  and
      Pyysalo, Sampo  and
      Miwa, Makoto  and
      Kim, Jin-Dong  and
      Tsujii, Jun{'}ichi",
    booktitle = "Proceedings of the 2010 Workshop on Biomedical Natural Language Processing",
    month = jul,
    year = "2010",
    address = "Uppsala, Sweden",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W10-1903",
    pages = "19--27",
}
"""

_DATASETNAME = "genia_ptm_event_corpus"

_DESCRIPTION = """\
Post-translational-modiﬁcations (PTM), amino acid modiﬁcations of proteins after translation, are one of the posterior
processes of protein biosynthesis for many proteins, and they are critical for determining protein function such as its
activity state, localization, turnover and interactions with other biomolecules. While there have been many studies of
information extraction targeting individual PTM types, there was until recently little effort to address extraction of
multiple PTM types at once in a unified framework.
"""

_HOMEPAGE = "http://www.geniaproject.org/other-corpora/ptm-event-corpus"

_LICENSE = "GENIA Project License for Annotated Corpora"

_URLS = {
    _DATASETNAME: "http://www.geniaproject.org/other-corpora/ptm-event-corpus/post-translational_modifications_training_data.tar.gz?attredirects=0&d=1",
}

_SUPPORTED_TASKS = [Tasks.EVENT_EXTRACTION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class GeniaPtmEventCorpusDataset(datasets.GeneratorBasedBuilder):
    """GENIA PTM event corpus."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="genia_ptm_event_corpus_source",
            version=SOURCE_VERSION,
            description="genia_ptm_event_corpus source schema",
            schema="source",
            subset_id="genia_ptm_event_corpus",
        ),
        BigBioConfig(
            name="genia_ptm_event_corpus_bigbio_kb",
            version=BIGBIO_VERSION,
            description="genia_ptm_event_corpus BigBio schema",
            schema="bigbio_kb",
            subset_id="genia_ptm_event_corpus",
        ),
    ]

    DEFAULT_CONFIG_NAME = "genia_ptm_event_corpus_source"

    def _info(self) -> datasets.DatasetInfo:
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
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),  # refers to the text_bound_annotation of the trigger
                            "trigger": datasets.Value("string"),
                            "arguments": [
                                {
                                    "role": datasets.Value("string"),
                                    "ref_id": datasets.Value("string"),
                                }
                            ],
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

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_dir": data_dir,
                },
            ),
        ]

    def _generate_examples(self, data_dir) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        for dirpath, _, filenames in os.walk(data_dir):
            for guid, filename in enumerate(filenames):
                if filename.endswith(".txt"):
                    txt_file_path = Path(dirpath, filename)
                    if self.config.schema == "source":
                        example = parsing.parse_brat_file(
                            txt_file_path, annotation_file_suffixes=[".a1", ".a2"]
                        )
                        example["id"] = str(guid)
                        for key in ["attributes", "normalizations"]:
                            del example[key]
                        yield guid, example
                    elif self.config.schema == "bigbio_kb":
                        example = parsing.brat_parse_to_bigbio_kb(
                            parsing.parse_brat_file(txt_file_path)
                        )
                        example["id"] = str(guid)
                        yield guid, example
