# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and
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

"""

from pathlib import Path
from typing import List

import datasets

from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks
from utils.parsing import brat_parse_to_bigbio_kb, parse_brat_file

_CITATION = """\
@inproceedings{kim-etal-2009-overview,
    title = "Overview of {B}io{NLP}{'}09 Shared Task on Event Extraction",
    author = "Kim, Jin-Dong  and
      Ohta, Tomoko  and
      Pyysalo, Sampo  and
      Kano, Yoshinobu  and
      Tsujii, Jun{'}ichi",
    booktitle = "Proceedings of the {B}io{NLP} 2009 Workshop Companion Volume for Shared Task",
    month = jun,
    year = "2009",
    address = "Boulder, Colorado",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W09-1401",
    pages = "1--9",
}
"""

_DATASETNAME = "bionlp_shared_task_2009"

_DESCRIPTION = """\
The BioNLP Shared Task 2009 was organized by GENIA Project and its corpora were curated based
on the annotations of the publicly available GENIA Event corpus and an unreleased (blind) section
of the GENIA Event corpus annotations, used for evaluation.
"""

_HOMEPAGE = "http://www.geniaproject.org/shared-tasks/bionlp-shared-task-2009"

_LICENSE = """
GENIA Project License for Annotated Corpora

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
- JST (Japan Science and Technology Agency) : http://www.jst.go.jp
"""


_URL_BASE = "http://www.nactem.ac.uk/GENIA/current/Shared-tasks/BioNLP-ST-2009/"
_URLS = {
    _DATASETNAME: {
        "train": _URL_BASE + "bionlp09_shared_task_training_data_rev2.tar.gz",
        "test": _URL_BASE + "bionlp09_shared_task_test_data_without_gold_annotation.tar.gz",
        "dev": _URL_BASE + "bionlp09_shared_task_development_data_rev1.tar.gz",
    },
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.EVENT_EXTRACTION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"

# https://2011.bionlp-st.org/bionlp-shared-task-2011/genia-event-extraction-genia
_ENTITY_TYPES = ["Protein", "Entity"]


class BioNLPSharedTask2009(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="bionlp_shared_task_2009_source",
            version=SOURCE_VERSION,
            description="bionlp_shared_task_2009 source schema",
            schema="source",
            subset_id="bionlp_shared_task_2009",
        ),
        BigBioConfig(
            name="bionlp_shared_task_2009_bigbio_kb",
            version=BIGBIO_VERSION,
            description="bionlp_shared_task_2009 BigBio schema",
            schema="bigbio_kb",
            subset_id="bionlp_shared_task_2009",
        ),
    ]

    DEFAULT_CONFIG_NAME = "bionlp_shared_task_2009_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "document_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "text_bound_annotations": [
                        {
                            "id": datasets.Value("string"),
                            "offsets": [[datasets.Value("int64")]],
                            "text": [datasets.Value("string")],
                            "type": datasets.Value("string"),
                        }
                    ],
                    "events": [
                        {
                            "arguments": [
                                {
                                    "ref_id": datasets.Value("string"),
                                    "role": datasets.Value("string"),
                                }
                            ],
                            "id": datasets.Value("string"),
                            "trigger": datasets.Value("string"),
                            "type": datasets.Value("string"),
                        }
                    ],
                    "relations": [
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "arg1_id": datasets.Value("string"),
                            "arg2_id": datasets.Value("string"),
                            "normalized": [
                                {
                                    "db_name": datasets.Value("string"),
                                    "db_id": datasets.Value("string"),
                                }
                            ],
                        }
                    ],
                    "equivalences": [datasets.Value("string")],
                    "attributes": [datasets.Value("string")],
                    "normalizations": [datasets.Value("string")],
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
        urls = _URLS[_DATASETNAME]
        data_dir_train = dl_manager.download_and_extract(urls["train"])
        data_dir_test = dl_manager.download_and_extract(urls["test"])
        data_dir_dev = dl_manager.download_and_extract(urls["dev"])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir_train,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": data_dir_test,
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": data_dir_dev,
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath, split) -> (int, dict):

        filepath = Path(filepath)
        txt_files: List[Path] = [file for file in filepath.iterdir() if file.suffix == ".txt"]

        if self.config.schema == "source":
            for i, file in enumerate(txt_files):
                brat_content = parse_brat_file(file)
                yield i, brat_content

        elif self.config.schema == "bigbio_kb":
            for i, file in enumerate(txt_files):
                brat_content = parse_brat_file(file)
                kb_example = brat_parse_to_bigbio_kb(brat_content, _ENTITY_TYPES)
                kb_example["id"] = kb_example["document_id"]
                yield i, kb_example
