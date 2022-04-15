# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and Ayush Singh (singhay).
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
import zipfile
from collections import defaultdict
from typing import List, Tuple, Dict

import datasets
from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

# https://dblp.org/rec/journals/jamia/HenryBFSU20.html?view=bibtex
_CITATION = """\
@article{DBLP:journals/jamia/HenryBFSU20,
  author    = {
                Sam Henry and
                Kevin Buchan and
                Michele Filannino and
                Amber Stubbs and
                Ozlem Uzuner
               },
  title     = {2018 n2c2 shared task on adverse drug events and medication extraction
               in electronic health records},
  journal   = {J. Am. Medical Informatics Assoc.},
  volume    = {27},
  number    = {1},
  pages     = {3--12},
  year      = {2020},
  url       = {https://doi.org/10.1093/jamia/ocz166},
  doi       = {10.1093/jamia/ocz166},
  timestamp = {Sat, 30 May 2020 19:53:56 +0200},
  biburl    = {https://dblp.org/rec/journals/jamia/HenryBFSU20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_DATASETNAME = "n2c2_2018_track2"

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
This dataset is designed for XXX NLP task.
"""

_HOMEPAGE = "https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/"

_LICENSE = "External Data User Agreement"

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"  # 2018-09-10

_BIGBIO_VERSION = "1.0.0"
C_PATTERN = r"c=\"(.+?)\" (\d+):(\d+) (\d+):(\d+)"
T_PATTERN = r"t=\"(.+?)\""
A_PATTERN = r"a=\"(.+?)\""
R_PATTERN = r"r=\"(.+?)\""

# Constants
DELIMITER = "||"
SOURCE = "source"
BIGBIO_KB = "bigbio_kb"
ANNOTATIONS_EXT = "ann"
TEXT_EXT = "txt"
N2C2_2018_NER_LABELS = sorted(['Drug', 'Frequency', 'Reason', 'ADE', 'Dosage', 'Duration', 'Form', 'Route', 'Strength'])
N2C2_2018_RELATION_LABELS = sorted(['Frequency-Drug', 'Strength-Drug', 'Route-Drug', 'Dosage-Drug', 'ADE-Drug',
                                    'Reason-Drug', 'Duration-Drug', 'Form-Drug'])


def _form_id(sample_id, entity_id, split, start_token, end_token, entity_type='entity'):
    return "{}-{}-{}-{}-{}-{}".format(
        sample_id,
        entity_type,
        entity_id,
        split,
        start_token,
        end_token,
    )


def _build_concept_dict(tag_id, tag_start, tag_end, tag_type, tag_text):
    return {
        "id": tag_id,
        "text": tag_text,
        "start": int(tag_start),
        "end": int(tag_end),
        "concept": tag_type,
    }


def _build_relation_dict(rel_id, arg1, arg2, rel_type):
    return {
        "id": rel_id,
        "arg1_id": arg1,
        "arg2_id": arg2,
        "relation": rel_type,
    }


def _get_annotations(annotation_file):
    """Return a dictionary with all the annotations in the .ann file."""
    tags, relations = {}, {}
    lines = annotation_file.splitlines()
    for line_num, line in enumerate(lines):
        if line.strip().startswith('T'):
            try:
                tag_id, tag_m, tag_text = line.strip().split('\t')
            except ValueError:
                print(line)

            if len(tag_m.split(' ')) == 3:
                tag_type, tag_start, tag_end = tag_m.split(' ')
            elif len(tag_m.split(' ')) == 4:
                tag_type, tag_start, _, tag_end = tag_m.split(' ')
            elif len(tag_m.split(' ')) == 5:
                tag_type, tag_start, _, _, tag_end = tag_m.split(' ')
            else:
                print(line)
            tags[tag_id] = _build_concept_dict(tag_id,
                                               tag_start,
                                               tag_end,
                                               tag_type,
                                               tag_text)

    for line_num, line in enumerate(filter(lambda line: line.strip().startswith('R'), lines)):
        rel_id, rel_m = line.strip().split('\t')
        rel_type, rel_arg1, rel_arg2 = rel_m.split(' ')
        rel_arg1 = rel_arg1.split(':')[1]
        rel_arg2 = rel_arg2.split(':')[1]
        arg1 = tags[rel_arg1]['id']
        arg2 = tags[rel_arg2]['id']
        relations[rel_id] = _build_relation_dict(rel_id, arg1, arg2, rel_type)

    return tags.values(), relations.values()


def _read_zip(file_path):
    samples = defaultdict(dict)
    with zipfile.ZipFile(file_path) as zf:
        for info in zf.infolist():

            base, filename = os.path.split(info.filename)
            _, ext = os.path.splitext(filename)
            ext = ext[1:]  # get rid of dot
            sample_id = filename.split(".")[0]

            if ext in [TEXT_EXT, ANNOTATIONS_EXT] and not filename.startswith("."):
                content = zf.read(info).decode("utf-8")
                if ext == TEXT_EXT:
                    samples[sample_id][ext] = content
                else:
                    samples[sample_id]["tags"], samples[sample_id]["relations"] = _get_annotations(content)

    return samples


def _get_entities_from_sample(sample_id, sample, split):
    entities = []
    text = sample[TEXT_EXT]
    for entity in sample['tags']:
        text_slice = text[entity["start"]: entity["end"]]
        text_slice_norm_1 = text_slice.replace("\n", "").lower()
        text_slice_norm_2 = text_slice.replace("\n", " ").lower()
        match = text_slice_norm_1 == entity["text"] or text_slice_norm_2 == entity["text"]
        if not match:
            continue
        entities.append({
            "id": _form_id(sample_id, entity["id"], split,
                           entity["start"], entity["end"]),
            "type": entity["concept"],
            "text": [text_slice],
            "offsets": [(entity["start"], entity["end"])],
            "normalized": [],
        })

    return entities


def _get_relations_from_sample(sample_id, sample, split):
    relations = []
    for relation in sample['relations']:
        relations.append({
            "id": _form_id(sample_id, relation["id"], split,
                           relation["arg1_id"], relation["arg2_id"], 'relation'),
            "type": relation["relation"],
            "arg1_id": relation["arg1_id"],
            "arg2_id": relation["arg2_id"],
            "normalized": [],
        })

    return relations


class N2C2AdverseDrugEventsMedicationExtractionDataset(datasets.GeneratorBasedBuilder):
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

    SOURCE_CONFIG_NAME = _DATASETNAME + "_" + SOURCE
    BIGBIO_CONFIG_NAME = _DATASETNAME + "_" + BIGBIO_KB

    BUILDER_CONFIGS = [
        BigBioConfig(
            name=SOURCE_CONFIG_NAME,
            version=SOURCE_VERSION,
            description=_DATASETNAME + " source schema",
            schema=SOURCE,
            subset_id=_DATASETNAME,
        ),
        BigBioConfig(
            name=BIGBIO_CONFIG_NAME,
            version=BIGBIO_VERSION,
            description=_DATASETNAME + " BigBio schema",
            schema=BIGBIO_KB,
            subset_id=_DATASETNAME,
        ),
    ]

    DEFAULT_CONFIG_NAME = SOURCE_CONFIG_NAME

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == SOURCE:
            features = datasets.Features(
                {
                    "doc_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "tags": [
                        {
                            "id": datasets.Value("string"),
                            "text": datasets.Value("string"),
                            "start": datasets.Value("int64"),
                            "end": datasets.Value("int64"),
                            "concept": datasets.ClassLabel(names=N2C2_2018_NER_LABELS),
                        }
                    ],
                    "relations": [
                        {
                            "id": datasets.Value("string"),
                            "arg1_id": datasets.Value("string"),
                            "arg2_id": datasets.Value("string"),
                            "relation": datasets.ClassLabel(names=N2C2_2018_RELATION_LABELS),
                        }
                    ]
                }
            )

        elif self.config.schema == BIGBIO_KB:
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
        if self.config.data_dir is None or self.config.name is None:
            raise ValueError("This is a local dataset. Please pass the data_dir and name kwarg to load_dataset.")
        else:
            data_dir = self.config.data_dir

        # Not all datasets have predefined canonical train/val/test splits.
        # If your dataset has no predefined splits, use datasets.Split.TRAIN for all of the data.

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "file_path": os.path.join(data_dir, "training_20180910.zip"),
                    "split": datasets.Split.TRAIN,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "file_path": os.path.join(data_dir, "gold-standard-test-data.zip"),
                    "split": datasets.Split.TEST,
                },
            )
        ]

    @staticmethod
    def _get_source_sample(sample_id, sample):
        return {
            "doc_id": sample_id,
            "text": sample.get(TEXT_EXT, ""),
            "tags": sample.get("tags", []),
            "relations": sample.get("relations", [])
        }

    @staticmethod
    def _get_bigbio_sample(sample_id, sample, split) -> dict:

        passage_text = sample.get("txt", "")
        entities = _get_entities_from_sample(sample_id, sample, split)
        relations = _get_relations_from_sample(sample_id, sample, split)
        return {
            "id": sample_id,
            "document_id": sample_id,
            "passages": [
                {
                    "id": f"{sample_id}-passage-0",
                    "type": "discharge summary",
                    "text": [passage_text],
                    "offsets": [(0, len(passage_text))],
                }
            ],
            "entities": entities,
            "relations": relations,
            "events": [],
            "coreferences": [],
        }

    def _generate_examples(self, file_path, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        samples = _read_zip(file_path)

        _id = 0

        for sample_id, sample in samples.items():

            if self.config.name == N2C2AdverseDrugEventsMedicationExtractionDataset.SOURCE_CONFIG_NAME:
                yield _id, self._get_source_sample(sample_id, sample)
            elif self.config.name == N2C2AdverseDrugEventsMedicationExtractionDataset.BIGBIO_CONFIG_NAME:
                yield _id, self._get_bigbio_sample(sample_id, sample, split)

            _id += 1


# TODO: Remove this before making your PR
if __name__ == "__main__":
    datasets.load_dataset(__file__,
                          name=N2C2AdverseDrugEventsMedicationExtractionDataset.BIGBIO_CONFIG_NAME,
                          data_dir="/Users/ayush.singh/workspace/data/Healthcare/i2b2_2018_ADE_Medication")
