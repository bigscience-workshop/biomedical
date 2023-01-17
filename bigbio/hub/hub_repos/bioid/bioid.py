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
from typing import Dict, Iterator, List, Tuple

import bioc
import datasets
import pandas as pd

# TODO : import the schema that fits your dataset:
from .bigbiohub import BigBioConfig, Tasks, kb_features

# TODO: add True or False boolean value indicating if this dataset is local or not
_LOCAL = False
_PUBMED = True
_LANGUAGES = ["English"]

# TODO: Add BibTeX citation
_CITATION = """\
@inproceedings{arighi2017bio,
  title={Bio-ID track overview},
  author={Arighi, Cecilia and Hirschman, Lynette and Lemberger, Thomas and Bayer, Samuel and Liechti, Robin and Comeau, Donald and Wu, Cathy},
  booktitle={Proc. BioCreative Workshop},
  volume={482},
  pages={376},
  year={2017}
}
"""

# TODO: create a module level variable with your dataset name (should match script name)
#  E.g. Hallmarks of Cancer: [dataset_name] --> hallmarks_of_cancer
_DATASETNAME = "bioid"
_DISPLAYNAME = "BIOID"

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
The Bio-ID track focuses on entity tagging and ID assignment to selected bioentity types.
The task is to annotate text from figure legends with the entity types and IDs for taxon (organism), gene, protein, miRNA, small molecules,
cellular components, cell types and cell lines, tissues and organs. The track draws on SourceData annotated figure
legends (by panel), in BioC format, and the corresponding full text articles (also BioC format) provided for context.
"""

# TODO: Add a link to an official homepage for the dataset here (if possible)
_HOMEPAGE = "https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vi/track-1/"

# TODO: Add the licence for the dataset here (if possible)
# Note that this doesn't have to be a common open source license.
# Some datasets have custom licenses. In this case, simply put the full license terms
# into `_LICENSE`
_LICENSE = "UNKNOWN"

# TODO: Add links to the urls needed to download your dataset files.
#  For local datasets, this variable can be an empty dictionary.

# For publicly available datasets you will most likely end up passing these URLs to dl_manager in _split_generators.
# In most cases the URLs will be the same for the source and bigbio config.
# However, if you need to access different files for each config you can have multiple entries in this dict.
# This can be an arbitrarily nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    _DATASETNAME: "https://biocreative.bioinformatics.udel.edu/media/store/files/2017/BioIDtraining_2.tar.gz",
}

_SUPPORTED_TASKS = [
    Tasks.NAMED_ENTITY_RECOGNITION,
    Tasks.NAMED_ENTITY_DISAMBIGUATION,
]

# TODO: set this to a version that is associated with the dataset. if none exists use "1.0.0"
#  This version doesn't have to be consistent with semantic versioning. Anything that is
#  provided by the original dataset as a version goes.
_SOURCE_VERSION = "2.0.0"

_BIGBIO_VERSION = "1.0.0"


# TODO: Name the dataset class to match the script name using CamelCase instead of snake_case
#  Append "Dataset" to the class name: BioASQ --> BioasqDataset
class BioidDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

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
            name="bioid_source",
            version=SOURCE_VERSION,
            description="bioid source schema",
            schema="source",
            subset_id="bioid",
        ),
        BigBioConfig(
            name="bioid_bigbio_kb",
            version=BIGBIO_VERSION,
            description="bioid BigBio schema",
            schema="bigbio_kb",
            subset_id="bioid",
        ),
    ]

    DEFAULT_CONFIG_NAME = "bioid_source"

    ENTITY_TYPES_NOT_NORMALIZAD = [
        "cell",
        "gene",
        "molecule",
        "protein",
        "subcellular",
        "tissue",
    ]

    def _info(self) -> datasets.DatasetInfo:

        # Create the source schema; this schema will keep all keys/information/labels as close to the original dataset as possible.
        # You can arbitrarily nest lists and dictionaries.
        # For iterables, use lists over tuples or `datasets.Sequence`
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "sourcedata_document": datasets.Value("string"),
                    "doi": datasets.Value("string"),
                    "pmc_id": datasets.Value("string"),
                    "figure": datasets.Value("string"),
                    "sourcedata_figure_dir": datasets.Value("string"),
                    "passages": [
                        {
                            "text": datasets.Value("string"),
                            "offset": datasets.Value("int32"),
                            "annotations": [
                                {
                                    "thomas_article": datasets.Value("string"),
                                    "doi": datasets.Value("string"),
                                    "don_article": datasets.Value("int32"),
                                    "figure": datasets.Value("string"),
                                    "annot id": datasets.Value("int32"),
                                    "paper id": datasets.Value("int32"),
                                    "first left": datasets.Value("int32"),
                                    "last right": datasets.Value("int32"),
                                    "length": datasets.Value("int32"),
                                    "byte length": datasets.Value("int32"),
                                    "left alphanum": datasets.Value("string"),
                                    "text": datasets.Value("string"),
                                    "right alphanum": datasets.Value("string"),
                                    "obj": datasets.Value("string"),
                                    "overlap": datasets.Value("string"),
                                    "identical span": datasets.Value("string"),
                                    "overlap_label_count": datasets.Value("int32"),
                                }
                            ],
                        }
                    ],
                }
            )

        # Choose the appropriate bigbio schema for your task and copy it here. You can find information on the schemas in the CONTRIBUTING guide.
        # In rare cases you may get a dataset that supports multiple tasks requiring multiple schemas. In that case you can define multiple bigbio configs with a bigbio_[bigbio_schema_name] format.
        # For example bigbio_kb, bigbio_t2t
        elif self.config.schema == "bigbio_kb":
            features = kb_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration

        # If you need to access the "source" or "bigbio" config choice, that will be in self.config.name

        # LOCAL DATASETS: You do not need the dl_manager; you can ignore this argument. Make sure `gen_kwargs` in the return gets passed the right filepath

        # PUBLIC DATASETS: Assign your data-dir based on the dl_manager.

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLs; many examples use the download_and_extract method; see the DownloadManager docs here: https://huggingface.co/docs/datasets/package_reference/builder_classes.html#datasets.DownloadManager

        # dl_manager can accept any type of nested list/dict and will give back the same structure with the url replaced with the path to local files.

        # TODO: KEEP if your dataset is PUBLIC; remove if not
        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        # Not all datasets have predefined canonical train/val/test splits.
        # If your dataset has no predefined splits, use datasets.Split.TRAIN for all of the data.

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    # TODO
                    "data_dir": data_dir,
                    "split": "train",
                },
            ),
        ]

    def load_annotations(self, path: str) -> Dict[str, Dict]:
        """
        We load annotations from `annotations.csv`
        becuase the one in the BioC xml files have offsets issues.
        """

        df = pd.read_csv(path, sep=",")

        df.fillna(-1, inplace=True)

        annotations: Dict[str, Dict] = {}

        for record in df.to_dict("records"):

            article_id = str(record["don_article"])

            if article_id not in annotations:
                annotations[article_id] = {}

            figure = record["figure"]

            if figure not in annotations:
                annotations[article_id][figure] = []

            annotations[article_id][figure].append(record)

        return annotations

    def load_data(self, data_dir: str) -> List[Dict]:
        """
        Compose text from BioC files with annotations from `annotations.csv`.
        We load annotations from `annotations.csv`
        becuase the one in the BioC xml files have offsets issues.
        """

        text_dir = os.path.join(data_dir, "BioIDtraining_2", "caption_bioc")
        annotation_file = os.path.join(data_dir, "BioIDtraining_2", "annotations.csv")

        annotations = self.load_annotations(path=annotation_file)

        data = []

        for file_name in os.listdir(text_dir):

            if file_name.startswith(".") or not file_name.endswith(".xml"):
                continue

            collection = bioc.load(os.path.join(text_dir, file_name))

            for document in collection.documents:

                item = document.infons

                assert (
                    len(document.passages) == 1
                ), "Document contains more than one passage (figure caption). This is not expected!"

                passage = document.passages[0]

                article_id = document.infons["pmc_id"]
                figure = document.infons["sourcedata_figure_dir"]

                try:
                    passage.annotations = annotations[article_id][figure]
                except KeyError:
                    passage.annotations = []

                item["passages"] = [
                    {
                        "text": passage.text,
                        "annotations": passage.annotations,
                        "offset": passage.offset,
                    }
                ]

                data.append(item)

        return data

    def get_entity(self, normalization: str) -> Tuple[str, List[Dict]]:
        """
        Compile normalization information from annotation
        """

        db_name_ids = normalization.split(":")

        db_ids = None

        # ids from cellosaurus do not have db name
        if len(db_name_ids) == 1:
            db_name = "Cellosaurus"
            db_ids = db_name_ids[0].split("|")
        else:
            # quirk
            if db_name_ids[0] == "CVCL_6412|CL":
                db_name = "Cellosaurus"
                db_ids = ["CVCL_6412"]
            else:
                db_name = db_name_ids[0]
                # db_name hints for entity type: skip if does not provide normalization
                if db_name not in self.ENTITY_TYPES_NOT_NORMALIZAD:
                    db_ids = [i.split(":")[1] for i in normalization.split("|")]

        normalized = (
            [{"db_name": db_name, "db_id": i} for i in db_ids]
            if db_ids is not None
            else []
        )

        # TODO: map to canonical entity types, ideally w/ a  dedicated enum like `Tasks`
        entity_type = db_name

        return entity_type, normalized

    def _generate_examples(
        self, data_dir: str, split: str
    ) -> Iterator[Tuple[int, Dict]]:
        """Yields examples as (key, example) tuples."""

        data = self.load_data(data_dir=data_dir)

        if self.config.schema == "source":
            for uid, document in enumerate(data):
                yield uid, document

        elif self.config.schema == "bigbio_kb":

            uid = 0  # global unique id

            for document in data:

                kb_document = {
                    "id": uid,
                    "document_id": document["pmc_id"],
                    "passages": [],
                    "entities": [],
                    "relations": [],
                    "events": [],
                    "coreferences": [],
                }

                uid += 1

                for passage in document["passages"]:
                    kb_document["passages"].append(
                        {
                            "id": uid,
                            "type": "figure_caption",
                            "text": [passage["text"]],
                            "offsets": [[0, len(passage["text"])]],
                        }
                    )
                    uid += 1

                    for a in passage["annotations"]:

                        entity_type, normalized = self.get_entity(a["obj"])

                        kb_document["entities"].append(
                            {
                                "id": uid,
                                "text": [a["text"]],
                                "type": entity_type,
                                "offsets": [[a["first left"], a["last right"]]],
                                "normalized": normalized,
                            }
                        )

                        uid += 1

                yield uid, kb_document
