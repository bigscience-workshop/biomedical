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
The main aim of MESINESP2 is to promote the development of practically relevant
semantic indexing tools for biomedical content in non-English language.
We have generated a manually annotated corpus, where domain experts have
labeled a set of scientific literature, clinical trials, and patent abstracts.
All the documents were labeled with DeCS descriptors, which is a structured controlled
vocabulary created by BIREME to index scientific publications on BvSalud, the largest
database of scientific documents in Spanish, which hosts records from the databases
LILACS, MEDLINE, IBECS, among others.

MESINESP track at BioASQ9 explores the efficiency of systems for assigning
DeCS to different types of biomedical documents. To that purpose, we have
divided the task into three subtracks depending on the document type. Then, for
each one we generated an annotated corpus which was provided to participating teams:

[Subtrack 1 corpus] MESINESP-L – Scientific Literature: It contains all
Spanish records from LILACS and IBECS databases at the Virtual Health Library
(VHL) with non-empty abstract written in Spanish.
[Subtrack 2 corpus] MESINESP-T- Clinical Trials contains records from Registro
Español de Estudios Clínicos (REEC). REEC doesn't provide documents with the
structure title/abstract needed in BioASQ, for that reason we have built
artificial abstracts based on the content available in the data crawled using the REEC API.
[Subtrack 3 corpus] MESINESP-P – Patents: This corpus includes patents in
Spanish extracted from Google Patents which have the IPC code “A61P” and “A61K31”.
In addition, we also provide a set of complementary data such as: the DeCS
terminology file, a silver standard with the participants' predictions to the task
background set and the entities of medications, diseases, symptoms and medical
procedures extracted from the BSC NERs documents.
"""

import json
import os
from typing import Dict, List, Tuple

import datasets

from biomed_datasets.utils import schemas
from biomed_datasets.utils.configs import BigBioConfig
from biomed_datasets.utils.constants import Tasks

_CITATION = """\
@conference {396,
    title = {Overview of BioASQ 2021-MESINESP track. Evaluation of
    advance hierarchical classification techniques for scientific
    literature, patents and clinical trials.},
    booktitle = {Proceedings of the 9th BioASQ Workshop
    A challenge on large-scale biomedical semantic indexing
    and question answering},
    year = {2021},
    url = {http://ceur-ws.org/Vol-2936/paper-11.pdf},
    author = {Gasco, Luis and Nentidis, Anastasios and Krithara, Anastasia
     and Estrada-Zavala, Darryl and Toshiyuki Murasaki, Renato and Primo-Pe{\~n}a,
     Elena and Bojo-Canales, Cristina and Paliouras, Georgios and Krallinger, Martin}
}

"""

_DATASETNAME = "bioasq_2021_mesinesp"


_DESCRIPTION = """
The main aim of MESINESP2 is to promote the development of practically relevant
semantic indexing tools for biomedical content in non-English language.
We have generated a manually annotated corpus, where domain experts have
labeled a set of scientific literature, clinical trials, and patent abstracts.
All the documents were labeled with DeCS descriptors, which is a structured controlled
vocabulary created by BIREME to index scientific publications on BvSalud, the largest
database of scientific documents in Spanish, which hosts records from the databases
LILACS, MEDLINE, IBECS, among others.

MESINESP track at BioASQ9 explores the efficiency of systems for assigning
DeCS to different types of biomedical documents. To that purpose, we have
divided the task into three subtracks depending on the document type. Then, for
each one we generated an annotated corpus which was provided to participating teams:

[Subtrack 1 corpus] MESINESP-L – Scientific Literature: It contains all
Spanish records from LILACS and IBECS databases at the Virtual Health Library
(VHL) with non-empty abstract written in Spanish.
[Subtrack 2 corpus] MESINESP-T- Clinical Trials contains records from Registro
Español de Estudios Clínicos (REEC). REEC doesn't provide documents with the
structure title/abstract needed in BioASQ, for that reason we have built
artificial abstracts based on the content available in the data crawled using the REEC API.
[Subtrack 3 corpus] MESINESP-P – Patents: This corpus includes patents in
Spanish extracted from Google Patents which have the IPC code “A61P” and “A61K31”.
In addition, we also provide a set of complementary data such as: the DeCS
terminology file, a silver standard with the participants' predictions to the task
background set and the entities of medications, diseases, symptoms and medical
procedures extracted from the BSC NERs documents.
"""

_HOMEPAGE = "https://zenodo.org/record/5602914#.YhSXJ5PMKWt"


_LICENSE = "CC-BY-4.0"

_URLS = {
    _DATASETNAME: {
        "subtrack1": "https://zenodo.org/record/5602914/files/Subtrack1-Scientific_Literature.zip?download=1",
        "subtrack2": "https://zenodo.org/record/5602914/files/Subtrack2-Clinical_Trials.zip?download=1",
        "subtrack3": "https://zenodo.org/record/5602914/files/Subtrack3-Patents.zip?download=1",
    },
}


_SUPPORTED_TASKS = [Tasks.TEXT_CLASSIFICATION]

_SOURCE_VERSION = "1.0.6"

_BIGBIO_VERSION = "1.0.0"


class Bioasq2021MesinespDataset(datasets.GeneratorBasedBuilder):
    """A dataset to promote the development of practically relevant
    semantic indexing tools for biomedical content in non-English language."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="bioasq_2021_mesinesp_subtrack1_all_source",
            version=SOURCE_VERSION,
            description="bioasq_2021_mesinesp source schema subtrack1",
            schema="source",
            subset_id="bioasq_2021_mesinesp_subtrack1_all",
        ),
        BigBioConfig(
            name="bioasq_2021_mesinesp_subtrack1_only_articles_source",
            version=SOURCE_VERSION,
            description="bioasq_2021_mesinesp source schema subtrack1",
            schema="source",
            subset_id="bioasq_2021_mesinesp_subtrack1_only_articles",
        ),
        BigBioConfig(
            name="bioasq_2021_mesinesp_subtrack2_source",
            version=SOURCE_VERSION,
            description="bioasq_2021_mesinesp source schema subtrack2",
            schema="source",
            subset_id="bioasq_2021_mesinesp_subtrack2",
        ),
        BigBioConfig(
            name="bioasq_2021_mesinesp_subtrack3_source",
            version=SOURCE_VERSION,
            description="bioasq_2021_mesinesp source schema subtrack3",
            schema="source",
            subset_id="bioasq_2021_mesinesp_subtrack3",
        ),
        BigBioConfig(
            name="bioasq_2021_mesinesp_subtrack1_all_bigbio_text",
            version=BIGBIO_VERSION,
            description="bioasq_2021_mesinesp BigBio schema subtrack1",
            schema="bigbio_text",
            subset_id="bioasq_2021_mesinesp_subtrack1_all",
        ),
        BigBioConfig(
            name="bioasq_2021_mesinesp_subtrack1_only_articles_bigbio_text",
            version=BIGBIO_VERSION,
            description="bioasq_2021_mesinesp BigBio schema subtrack1",
            schema="bigbio_text",
            subset_id="bioasq_2021_mesinesp_subtrack1_only_articles",
        ),
        BigBioConfig(
            name="bioasq_2021_mesinesp_subtrack2_bigbio_text",
            version=BIGBIO_VERSION,
            description="bioasq_2021_mesinesp BigBio schema subtrack2",
            schema="bigbio_text",
            subset_id="bioasq_2021_mesinesp_subtrack2",
        ),
        BigBioConfig(
            name="bioasq_2021_mesinesp_subtrack3_bigbio_text",
            version=BIGBIO_VERSION,
            description="bioasq_2021_mesinesp BigBio schema subtrack3",
            schema="bigbio_text",
            subset_id="bioasq_2021_mesinesp_subtrack3",
        ),
    ]

    DEFAULT_CONFIG_NAME = "bioasq_2021_mesinesp_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "abstractText": datasets.Value("string"),
                    "db": datasets.Value("string"),
                    "decsCodes": datasets.Sequence(datasets.Value("string")),
                    "id": datasets.Value("string"),
                    "journal": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "year": datasets.Value("string"),
                }
            )
        elif self.config.schema == "bigbio_text":
            features = schemas.text.features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        if "subtrack1" in self.config.name:
            track = "1"
        elif "subtrack2" in self.config.name:
            track = "2"
        else:
            track = "3"

        urls = _URLS[_DATASETNAME][f"subtrack{track}"]
        if self.config.data_dir is None:
            try:
                data_dir = dl_manager.download_and_extract(urls)
            except ConnectionError:
                raise ConnectionError(
                    "Could not download. Save locally and use `data_dir` kwarg"
                )
        else:
            data_dir = self.config.data_dir

        if track == "1":
            top_folder = "Subtrack1-Scientific_Literature"
        elif track == "2":
            top_folder = "Subtrack2-Clinical_Trials"
        else:
            top_folder = "Subtrack3-Patents"
        if track == "1":
            if "all" in self.config.name:
                train_filepath = "training_set_subtrack1_all.json"
            else:
                train_filepath = "training_set_subtrack1_only_articles.json"
        else:
            train_filepath = f"training_set_subtrack{track}.json"

        dev_filepath = f"development_set_subtrack{track}.json"
        test_filepath = f"test_set_subtrack{track}.json"

        split_gens = [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, top_folder, "Train", train_filepath
                    ),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, top_folder, "Development", dev_filepath
                    ),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, top_folder, "Test", test_filepath
                    ),
                },
            ),
        ]

        # track 3 doesn't have Train data
        if track == "3":
            return split_gens[1:]

        return split_gens

    def _generate_examples(self, filepath) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        if self.config.schema == "source":

            with open(filepath) as fp:
                data = json.load(fp)

            for key, example in enumerate(data["articles"]):
                yield key, example

        elif self.config.schema == "bigbio_text":
            with open(filepath) as fp:
                data = json.load(fp)

            for key, example in enumerate(data["articles"]):
                yield key, {
                    "id": example["id"],
                    "document_id": "NULL",
                    "text": example["abstractText"],
                    "labels": example["decsCodes"],
                }
