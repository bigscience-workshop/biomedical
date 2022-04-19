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
from glob import glob
import re
from typing import List, Tuple, Dict

from bs4 import BeautifulSoup
import datasets
from yaml import parse
from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

_CITATION = """\
@article{bunescu2005comparitive,
  title={Comparative experiments on learning information extractors for proteins and their interactions.},
  author={Bunescu, Razvan et al.},
  journal={Artificial intelligence in medicine},
  volume={33},
  number={2},
  pages={139-155},
  year={2005}
}
"""


# TODO: create a module level variable with your dataset name (should match script name)
#  E.g. Hallmarks of Cancer: [dataset_name] --> hallmarks_of_cancer
_DATASETNAME = "aimed"

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
This dataset is designed for the NLP task of NER.
"""

# TODO: Add a link to an official homepage for the dataset here (if possible)
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here (if possible)
# Note that this doesn't have to be a common open source license.
# Some datasets have custom licenses. In this case, simply put the full license terms
# into `_LICENSE`
_LICENSE = ""  # TODO: NULL

# TODO: Add links to the urls needed to download your dataset files.
#  For local datasets, this variable can be an empty dictionary.

# For publicly available datasets you will most likely end up passing these URLs to dl_manager in _split_generators.
# In most cases the URLs will be the same for the source and bigbio config.
# However, if you need to access different files for each config you can have multiple entries in this dict.
# This can be an arbitrarily nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    _DATASETNAME: [
        "https://www.cs.utexas.edu/ftp/mooney/bio-data/interactions.tar.gz",
        "https://www.cs.utexas.edu/ftp/mooney/bio-data/proteins.tar.gz",
    ],
}

# TODO: add supported task by dataset. One dataset may support multiple tasks
_SUPPORTED_TASKS = [
    Tasks.RELATION_EXTRACTION
]  # example: [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

# TODO: set this to a version that is associated with the dataset. if none exists use "1.0.0"
#  This version doesn't have to be consistent with semantic versioning. Anything that is
#  provided by the original dataset as a version goes.
_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


def remove_page_numbers(text):
    text = re.sub("PG - \d+ - \d+ ", "", text)
    return text


def parse_interaction_abstracts(fpaths):
    all_entries = []
    count = 0
    passage_count = 0
    relation_count = 0
    for a in fpaths:
        # Placeholders to build up later
        str_chunks = []
        prot_chunks = []
        prot_text = (
            set()
        )  # Used to keep <prot> chunks and text in descendants from being repeated
        pairs = {}
        p1p2 = []
        current = 0

        # Start constructing parsed entry for this file
        entry = {
            "id": count,
            "document_id": count,
            "passages": [],
            "entities": [],
            "relations": [],
            "events": [],
            "coreferences": [],
        }
        count += 1

        with open(a, "r") as f:
            content = f.read().replace("\n", " ")
        soup = BeautifulSoup(content, "html.parser")

        for chunk in soup.descendants:
            # Handle proteins
            if chunk.name == "prot":
                text = chunk.text
                prot_chunks.append(chunk)
                prot_text.update({text})
                str_chunks.append(text)

                # Construct entity
                entry["entities"].append(
                    {
                        "id": id(chunk),
                        "type": "protein",
                        "text": [text],
                        "offsets": [[current, current + len(text)]],
                        "normalized": [],
                    }
                )

                current += len(text)

            # Handle string chunks, which also include inner text from <prot>, <p1>, and <p2> tags
            if isinstance(chunk, str):
                # Adding text directly from <prot> tags lets us track offsets more reliably
                # So we basically keep pushing/popping from a set as we come across <prot> tags and their associated strings
                if chunk in prot_text:
                    prot_text.discard(chunk)
                else:
                    # clean page numbers, may as well do it here since they never appear in prot tags
                    text = remove_page_numbers(chunk)
                    str_chunks.append(text)
                    current += len(text)

            # Handle interactions
            if chunk.name in ["p1", "p2"]:
                pair = chunk.get("pair")
                name = chunk.name

                # May have multiple prots in an interaction, I chose to represent all
                chunk_prots = chunk.select("prot")
                chunk_prots = [
                    {"id": id(prot), "prot": prot, "text": prot.text}
                    for prot in chunk_prots
                ]

                parsed = {
                    "id": id(chunk),
                    "prots": chunk_prots,
                    "pair": chunk.get("pair"),
                    "name": chunk.name,
                    "text": chunk.text.strip(),
                }

                p1p2.append(parsed)
                if pair not in pairs:
                    pairs[pair] = dict()
                pairs[pair][name] = parsed

        # Construct cleaned text
        clean = "".join(str_chunks)

        # Test entities
        for e in entry["entities"]:
            start, end = e["offsets"][0]
            assert e["text"][0] == clean[start:end]

        # Construct interactions
        for n, pair in pairs.items():
            p1 = pair.get("p1")
            p2 = pair.get("p2")
            if not (p1 and p2):
                break
            for p1_prot in p1["prots"]:
                for p2_prot in p2["prots"]:
                    relation = {
                        "id": relation_count,
                        "type": "protein interaction",
                        "arg1_id": p1_prot["id"],
                        "arg2_id": p2_prot["id"],
                        "normalized": []
                    }
                    entry["relations"].append(relation)
                    relation_count += 1

        # Extract passages
        title = re.search("TI - (.*?) AB -", clean)
        abstract = re.search("AB - (.*?) AD -", clean)
        ad = re.search("AB - .*? AD - (.*)$", clean)

        # Test that passages exist
        assert title and abstract and ad

        # Construct passage list
        entry["passages"] = [
            {
                "id": "0",
                "type": "title",
                "text": [title.group(1)],
                "offsets": [list(title.span(1))],
            },
            {
                "id": "1",
                "type": "abstract",
                "text": [abstract.group(1)],
                "offsets": [list(abstract.span(1))],
            },
            {
                "id": "2",
                "type": "AD",
                "text": [ad.group(1)],
                "offsets": [list(ad.span(1))],
            },
        ]

        # Test passage offsets
        for passage in entry["passages"]:
            start, end = passage["offsets"][0]
            assert clean[start:end] == passage["text"][0]

        all_entries.append(entry)

    # Post process entries to get better IDs for proteins
    entity_mapping = dict()
    entity_count = 0

    for entry in all_entries:
        for entity in entry["entities"]:
            entity_mapping[entity["id"]] = entity_count
            entity["id"] = entity_count
            entity_count += 1
        for relation in entry["relations"]:
            relation["arg1_id"] = entity_mapping[relation["arg1_id"]]
            relation["arg2_id"] = entity_mapping[relation["arg2_id"]]

    return all_entries


def parse_protein_abstracts(fpaths):
    all_entries = []
    passage_count = 0
    entity_count = 0

    for p in fpaths:
        entry = {
            "id": 0,
            "document_id": 0,
            "passages": [],
            "entities": [],
            "relations": [],
            "events": [],
            "coreferences": [],
        }

        with open(p, "r") as f:
            content = f.read().replace("\n", " ").strip()
        with open(p, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        # assert(lines[0].startswith('<ArticleTitle>'))
        # assert(lines[1].startswith('<AbstractText>'))

        title_line = lines[0]
        title_line = re.sub("</?prot>", "", title_line)
        title_line = re.sub("</?ArticleTitle>", "", title_line)

        abstract_line = lines[1]
        abstract_line = re.sub("</?prot>", "", abstract_line)
        abstract_line = re.sub("</?AbstractText>", "", abstract_line)

        title_offset = [[0, len(title_line)]]
        abstract_offset = [
            [len(title_line) + 1, 1 + len(title_line) + len(abstract_line)]
        ]

        title_passage = {
            "id": id(title_line),
            "type": "title",
            "text": [title_line],
            "offsets": title_offset,
        }

        abstract_passage = {
            "id": id(abstract_line),
            "type": "abstract",
            "text": [abstract_line],
            "offsets": abstract_offset,
        }

        entry["passages"] = [title_passage, abstract_passage]

        clean = title_line + " " + abstract_line

        soup = BeautifulSoup(content, "html.parser")
        str_chunks = []
        current = 0
        prot_str = set()
        prot_chunks = []

        for chunk in soup.descendants:

            # Handle proteins
            if chunk.name == "prot":
                text = chunk.text
                prot_chunks.append(chunk)
                prot_str.update({text})
                str_chunks.append(text)

                # Construct entity
                entry["entities"].append(
                    {
                        "id": id(chunk),
                        "type": "protein",
                        "text": [text],
                        "offsets": [[current, current + len(text)]],
                        "normalized": [],
                    }
                )

                current += len(text)

            # Handle string chunks, which also include inner text from <prot>, <p1>, and <p2> tags
            if isinstance(chunk, str):
                # Adding text directly from <prot> tags lets us track offsets more reliably
                # So we basically keep pushing/popping from a set as we come across <prot> tags and their associated strings
                if chunk in prot_str:
                    prot_str.discard(chunk)
                else:
                    # clean page numbers, may as well do it here since they never appear in prot tags
                    text = remove_page_numbers(chunk)
                    str_chunks.append(text)
                    current += len(text)

        all_entries.append(entry)

    return all_entries


# TODO: Name the dataset class to match the script name using CamelCase instead of snake_case
#  Append "Dataset" to the class name: BioASQ --> BioasqDataset
class AIMedDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="aimed_source",
            version=SOURCE_VERSION,
            description="AIMed source schema",
            schema="source",
            subset_id="aimed",
        ),
        BigBioConfig(
            name="aimed_bigbio_kb",
            version=BIGBIO_VERSION,
            description="AIMed BigBio schema",
            schema="bigbio_kb",
            subset_id="aimed",
        ),
    ]

    DEFAULT_CONFIG_NAME = "aimed_source"

    def _info(self) -> datasets.DatasetInfo:

        print()
        print(self.config.schema)
        print()

        if self.config.schema == "source":
            features = datasets.Features(
                {"doc_id": datasets.Value("string"), "xml": datasets.Value("string")}
            )

        # For example bigbio_kb, bigbio_t2t
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
        # If you need to access the "source" or "bigbio" config choice, that will be in self.config.name

        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)
        print(data_dir)

        if "interaction" in glob(data_dir[0] + "/*/**")[0]:
            interaction_dir, protein_dir = data_dir
        else:
            protein_dir, interaction_dir = data_dir

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "interaction_dir": interaction_dir,
                    "protein_dir": protein_dir,
                    "split": "train",
                },
            )
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`

    # TODO: change the args of this function to match the keys in `gen_kwargs`. You may add any necessary kwargs.

    def _generate_examples(
        self, interaction_dir: str, protein_dir: str, split: str
    ) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        interaction_fpaths = glob(interaction_dir + "/*/**")
        protein_fpaths = glob(protein_dir + "/*/**")

        if self.config.schema == "source":
            all_fpaths = interaction_fpaths + protein_fpaths

            for ix, fpath in enumerate(all_fpaths):
                key = ix
                with open(fpath, "r") as f:
                    xml = f.read()
                example = {"doc_id": ix, "xml": xml}
                yield key, example

        elif self.config.schema == "bigbio_kb":

            all_entries_interaction = parse_interaction_abstracts(interaction_fpaths)
            all_entries_protein = parse_protein_abstracts(protein_fpaths)
            all_entries = all_entries_interaction + all_entries_protein

            for key, example in enumerate(all_entries):
                yield key, example


# This template is based on the following template from the datasets package:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py
