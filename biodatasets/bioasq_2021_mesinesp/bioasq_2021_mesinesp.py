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
This template serves as a starting point for contributing a dataset to the BigScience Biomedical repo.

When modifying it for your dataset, look for TODO items that offer specific instructions.

Full documentation on writing dataset loading scripts can be found here:
https://huggingface.co/docs/datasets/add_dataset.html

To create a dataset loading script you will create a class and implement 3 methods:
  * `_info`: Establishes the schema for the dataset, and returns a datasets.DatasetInfo object.
  * `_split_generators`: Downloads and extracts data for each split (e.g. train/val/test) or associate local data with each split.
  * `_generate_examples`: Creates examples from data on disk that conform to each schema defined in `_info`.

TODO: Before submitting your script, delete this doc string and replace it with a description of your dataset.

[bigbio_schema_name] = (kb, pairs, qa, text, t2t, entailment)
"""

import os
import json
from typing import List, Tuple, Dict

import datasets
from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

# TODO: Add BibTeX citation
_CITATION = """\
@article{,
  author    = {},
  title     = {},
  journal   = {},
  volume    = {},
  year      = {},
  url       = {},
  doi       = {},
  biburl    = {},
  bibsource = {}
}
"""

_DATASETNAME = "bioasq_2021_mesinesp"

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
The main aim of MESINESP2 is to promote the development of practically relevant semantic indexing tools for biomedical content in non-English language. We have generated a manually annotated corpus, where domain experts have labeled a set of scientific literature, clinical trials, and patent abstracts. All the documents were labeled with DeCS descriptors, which is a structured controlled vocabulary created by BIREME to index scientific publications on BvSalud, the largest database of scientific documents in Spanish, which hosts records from the databases LILACS, MEDLINE, IBECS, among others. 

MESINESP track at BioASQ9 explores the efficiency of systems for assigning DeCS to different types of biomedical documents. To that purpose, we have divided the task into three subtracks depending on the document type. Then, for each one we generated an annotated corpus which was provided to participating teams:

[Subtrack 1 corpus] MESINESP-L – Scientific Literature: It contains all Spanish records from LILACS and IBECS databases at the Virtual Health Library (VHL) with non-empty abstract written in Spanish.
[Subtrack 2 corpus] MESINESP-T- Clinical Trials contains records from Registro Español de Estudios Clínicos (REEC). REEC doesn't provide documents with the structure title/abstract needed in BioASQ, for that reason we have built artificial abstracts based on the content available in the data crawled using the REEC API. 
[Subtrack 3 corpus] MESINESP-P – Patents: This corpus includes patents in Spanish extracted from Google Patents which have the IPC code “A61P” and “A61K31”.
In addition, we also provide a set of complementary data such as: the DeCS terminology file, a silver standard with the participants' predictions to the task background set and the entities of medications, diseases, symptoms and medical procedures extracted from the BSC NERs documents.
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
    """A dataset to promote the development of practically relevant semantic indexing tools for biomedical content in non-English language."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    # You will be able to load the "source" or "bigbio" configurations with
    # ds_source = datasets.load_dataset('my_dataset', name='source')
    # ds_bigbio = datasets.load_dataset('my_dataset', name='bigbio')

    # For local datasets you can make use of the `data_dir` and `data_files` kwargs
    # https://huggingface.co/docs/datasets/add_dataset.html#downloading-data-files-and-organizing-splits
    # ds_source = datasets.load_dataset('my_dataset', name='source', data_dir="/path/to/data/files")
    # ds_bigbio = datasets.load_dataset('my_dataset', name='bigbio', data_dir="/path/to/data/files")

    # TODO: For each dataset, implement Config for Source and BigBio;
    #  If dataset contains more than one subset (see examples/bioasq.py) implement for EACH of them.
    #  Each of them should contain:
    #   - name: should be unique for each dataset config eg. bioasq10b_(source|bigbio)_[bigbio_schema_name]
    #   - version: option = (SOURCE_VERSION|BIGBIO_VERSION)
    #   - description: one line description for the dataset
    #   - schema: options = (source|bigbio_[bigbio_schema_name])
    #   - subset_id: subset id is the canonical name for the dataset (eg. bioasq10b)
    #  where [bigbio_schema_name] = (kb, pairs, qa, text, t2t, entailment)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="bioasq_2021_mesinesp_subtrack1_all_source",
            version=SOURCE_VERSION,
            description="bioasq_2021_mesinesp source schema subtrack1",
            schema="source",
            subset_id="bioasq_2021_mesinesp",
        ),
        BigBioConfig(
            name="bioasq_2021_mesinesp_subtrack1_only_articles_source",
            version=SOURCE_VERSION,
            description="bioasq_2021_mesinesp source schema subtrack1",
            schema="source",
            subset_id="bioasq_2021_mesinesp",
        ),
        BigBioConfig(
            name="bioasq_2021_mesinesp_subtrack2_source",
            version=SOURCE_VERSION,
            description="bioasq_2021_mesinesp source schema subtrack2",
            schema="source",
            subset_id="bioasq_2021_mesinesp",
        ),
        BigBioConfig(
            name="bioasq_2021_mesinesp_subtrack3_source",
            version=SOURCE_VERSION,
            description="bioasq_2021_mesinesp source schema subtrack3",
            schema="source",
            subset_id="bioasq_2021_mesinesp",
        ),
        BigBioConfig(
            name="bioasq_2021_mesinesp_subtrack1_bigbio_text",
            version=BIGBIO_VERSION,
            description="bioasq_2021_mesinesp BigBio schema subtrack1",
            schema="bigbio_text",
            subset_id="bioasq_2021_mesinesp",
        ),
        BigBioConfig(
            name="bioasq_2021_mesinesp_subtrack2_bigbio_text",
            version=BIGBIO_VERSION,
            description="bioasq_2021_mesinesp BigBio schema subtrack2",
            schema="bigbio_text",
            subset_id="bioasq_2021_mesinesp",
        ),
        BigBioConfig(
            name="bioasq_2021_mesinesp_subtrack3_bigbio_text",
            version=BIGBIO_VERSION,
            description="bioasq_2021_mesinesp BigBio schema subtrack3",
            schema="bigbio_text",
            subset_id="bioasq_2021_mesinesp",
        ),
    ]

    DEFAULT_CONFIG_NAME = "bioasq_2021_mesinesp_source"

    def _info(self) -> datasets.DatasetInfo:

        # Create the source schema; this schema will keep all keys/information/labels as close to the original dataset as possible.

        # You can arbitrarily nest lists and dictionaries.
        # For iterables, use lists over tuples or `datasets.Sequence`

        if self.config.schema == "source":
                features = datasets.Features(
                    {
                        "articles": [{
                            "abstractText": datasets.Value("string"),
                            "db":datasets.Value("string"),
                            "decsCodes": datasets.Sequence(datasets.Value("string")),
                            "id":datasets.Value("string"),
                            "journal":datasets.Value("string"),
                            "title":datasets.Value("string"),
                            "year": datasets.Value("int32"),
                        }]
                    }
                )

        # Choose the appropriate bigbio schema for your task and copy it here. You can find information on the schemas in the CONTRIBUTING guide.

        # In rare cases you may get a dataset that supports multiple tasks requiring multiple schemas. In that case you can define multiple bigbio configs with a bigbio_[bigbio_schema_name] format.

        # For example bigbio_kb, bigbio_t2t
        elif self.config.schema == "bigbio_text":
            features = schemas.text

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

        urls = _URLS[_DATASETNAME[f"subtrack{track}"]]
        if self.config.data_dir is None:
            try:
                data_dir = dl_manager.download_and_extract(urls)
            except:
                raise ConnectionError("Could not download. Save locally and use `data_dir` kwarg")
        else:
            data_dir = self.config.data_dir

        if self.config.name == "bioasq_2021_mesinesp_subtrack1_all_source":
            train_filepath = "training_set_subtrack1_all.json"
        elif self.config.name == "bioasq_2021_mesinesp_subtrack1_only_articles_source" or self.config.schema == "bigbio_text":
            train_filepath = "training_set_subtrack1_only_articles.json"
        else:
            train_filepath = f"training_set_subtrack{track}.json"

        dev_filepath = f"development_set_subtrack{track}.json"
        test_filepath = f"testing_set_subtrack{track}.json"

        split_gens = [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "Train", train_filepath),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "Development", dev_filepath),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "Test", test_filepath),
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
                        "labels": example["descCodes"],
                }

if __name__ == "__main__":
    datasets.load_dataset(__file__)
