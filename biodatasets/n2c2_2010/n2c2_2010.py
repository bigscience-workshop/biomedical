# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and
#
# * Ayush Singh (singhay)
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
  * `_split_generators`: Downloads and extracts data for each split (e.g. train/val/test) or associate local data with
  each split.
  * `_generate_examples`: Creates examples from data on disk that conform to each schema defined in `_info`.

TODO: Before submitting your script, delete this doc string and replace it with a description of your dataset.

"""

import os
import re
import tarfile
from collections import defaultdict
from typing import List, Match

import datasets
from dataclasses import dataclass

from typing import Tuple

from datasets import Version

from utils import schemas
from utils.constants import Tasks

_CITATION = """\
@article{DBLP:journals/jamia/UzunerSSD11,
  author    = {
                Ozlem Uzuner and
                Brett R. South and
                Shuying Shen and
                Scott L. DuVall
               },
  title     = {2010 i2b2/VA challenge on concepts, assertions, and relations in clinical
               text},
  journal   = {J. Am. Medical Informatics Assoc.},
  volume    = {18},
  number    = {5},
  pages     = {552--556},
  year      = {2011},
  url       = {https://doi.org/10.1136/amiajnl-2011-000203},
  doi       = {10.1136/amiajnl-2011-000203},
  timestamp = {Mon, 11 May 2020 23:00:20 +0200},
  biburl    = {https://dblp.org/rec/journals/jamia/UzunerSSD11.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_DATASETNAME = "n2c2_2010"

_DESCRIPTION = """\
The i2b2/VA corpus contained de-identified discharge summaries from Beth Israel
Deaconess Medical Center, Partners Healthcare, and University of Pittsburgh Medical
Center (UPMC). In addition, UPMC contributed de-identified progress notes to the
i2b2/VA corpus. This dataset contains the records from Beth Israel and Partners.

The 2010 i2b2/VA Workshop on Natural Language Processing Challenges for Clinical Records comprises three tasks: 
1) a concept extraction task focused on the extraction of medical concepts from patient reports; 
2) an assertion classification task focused on assigning assertion types for medical problem concepts; 
3) a relation classification task focused on assigning relation types that hold between medical problems, 
tests, and treatments. 

i2b2 and the VA provided an annotated reference standard corpus for the three tasks. 
Using this reference standard, 22 systems were developed for concept extraction, 
21 for assertion classification, and 16 for relation classification.
"""

_HOMEPAGE = "https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/"

_LICENSE = "Data User Agreement"

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


def _read_tar_gz(file_path: str, samples=None):
    if samples is None:
        samples = defaultdict(dict)
    with tarfile.open(file_path, "r:gz") as tf:

        for member in tf.getmembers():
            base, filename = os.path.split(member.name)
            _, ext = os.path.splitext(filename)
            ext = ext[1:]  # get rid of dot
            sample_id = filename.split(".")[0]

            if ext in ["txt", "ast", "con", "rel"]:
                samples[sample_id][f"{ext}_source"] = os.path.basename(file_path) + "|" + member.name

                with tf.extractfile(member) as fp:
                    content_bytes = fp.read()

                content = content_bytes.decode("utf-8")
                samples[sample_id][ext] = content

    return samples


C_PATTERN = r"c=\"(.+?)\" (\d+):(\d+) (\d+):(\d+)"
T_PATTERN = r"t=\"(.+?)\""
A_PATTERN = r"a=\"(.+?)\""
R_PATTERN = r"r=\"(.+?)\""

# Constants
DELIMITER = "||"
SOURCE = "source"
BIGBIO_KB = "bigbio_kb"


def _ct_match_to_dict(c_match: Match, t_match: Match) -> dict:
    """Return a dictionary with groups from concept and type regex matches."""
    return {
        "text": c_match.group(1),
        "start_line": int(c_match.group(2)),
        "start_token": int(c_match.group(3)),
        "end_line": int(c_match.group(4)),
        "end_token": int(c_match.group(5)),
        "type": t_match.group(1),
    }


def _parse_con_line(line: str) -> dict:
    """Parse one line from a *.con file.

    A typical line has the form,
      'c="angie cm johnson , m.d." 13:2 13:6||t="person"

    This represents one concept to be placed into a coreference group.
    It can be interpreted as follows,
      'c="<string>" <start_line>:<start_token> <end_line>:<end_token>||t="<concept type>"'

    """
    c_part, t_part = line.split(DELIMITER)
    c_match, t_match = re.match(C_PATTERN, c_part), re.match(T_PATTERN, t_part)
    return {
        "text": c_match.group(1),
        "start_line": int(c_match.group(2)),
        "start_token": int(c_match.group(3)),
        "end_line": int(c_match.group(4)),
        "end_token": int(c_match.group(5)),
        "type": t_match.group(1),
    }


def _parse_rel_line(line: str) -> dict:
    """Parse one line from a *.rel file.

    A typical line has the form,
      'c="coronary artery bypass graft" 115:4 115:7||r="TrAP"||c="coronary artery disease" 115:0 115:2'

    This represents two concepts related to one another.
    It can be interpreted as follows,
      'c="<string>" <start_line>:<start_token> <end_line>:<end_token>||r="<type>"||c="<string>"
      <start_line>:<start_token> <end_line>:<end_token>'

    """
    c1_part, r_part, c2_part = line.split(DELIMITER)
    c1_match, r_match, c2_match = re.match(C_PATTERN, c1_part), re.match(R_PATTERN, r_part), re.match(C_PATTERN, c2_part)
    return {
        "concept_1": {
            "text": c1_match.group(1),
            "start_line": int(c1_match.group(2)),
            "start_token": int(c1_match.group(3)),
            "end_line": int(c1_match.group(4)),
            "end_token": int(c1_match.group(5)),
        },
        "concept_2": {
            "text": c2_match.group(1),
            "start_line": int(c2_match.group(2)),
            "start_token": int(c2_match.group(3)),
            "end_line": int(c2_match.group(4)),
            "end_token": int(c2_match.group(5)),
        },
        "relation": r_match.group(1),
    }


def _parse_ast_line(line: str) -> dict:
    """Parse one line from a *.ast file.

    A typical line has the form,
      'c="mild inferior wall hypokinesis" 42:2 42:5||t="problem"||a="present"'

    This represents one concept along with it's assertion.
    It can be interpreted as follows,
      'c="<string>" <start_line>:<start_token> <end_line>:<end_token>||t="<concept type>"||a="<assertion type>"'

    """
    c_part, t_part, a_part = line.split(DELIMITER)
    c_match, t_match, a_match = re.match(C_PATTERN, c_part), re.match(T_PATTERN, t_part), re.match(A_PATTERN, a_part)
    return {
        "text": c_match.group(1),
        "start_line": int(c_match.group(2)),
        "start_token": int(c_match.group(3)),
        "end_line": int(c_match.group(4)),
        "end_token": int(c_match.group(5)),
        "type": t_match.group(1),
        "assertion": a_match.group(1),
    }


def _tokoff_from_line(text: str) -> List[Tuple[int, int]]:
    """Produce character offsets for each token (whitespace split)

    For example,
      text = " one  two three ."
      tokoff = [(1,4), (6,9), (10,15), (16,17)]
    """
    tokoff = []
    start = None
    end = None
    for ii, char in enumerate(text):
        if char != " " and start is None:
            start = ii
        if char == " " and start is not None:
            end = ii
            tokoff.append((start, end))
            start = None
    if start is not None:
        end = ii + 1
        tokoff.append((start, end))
    return tokoff


def _form_entity_id(sample_id, split, start_line, start_token, end_line, end_token):
    return "{}-entity-{}-{}-{}-{}-{}".format(
        sample_id,
        split,
        start_line,
        start_token,
        end_line,
        end_token,
    )


def _get_relations_from_sample(sample_id, sample, split):
    rel_lines = sample["rel"].splitlines()

    relations = []
    for i, rel_line in enumerate(rel_lines):
        rel = _parse_rel_line(rel_line)
        rel_id = _form_entity_id(sample_id, split,
                                 rel['concept_1']['start_line'],
                                 rel['concept_1']['start_token'],
                                 rel['concept_1']['end_line'],
                                 rel['concept_1']['end_token'])
        rel_id += '-' + rel['relation'] + '-'
        rel_id += _form_entity_id(sample_id, split,
                                 rel['concept_2']['start_line'],
                                 rel['concept_2']['start_token'],
                                 rel['concept_2']['end_line'],
                                 rel['concept_2']['end_token'])
        rel.update({'id': rel_id})
        relations.append(rel)

    return relations


def _get_entities_from_sample(sample_id, sample, split):
    """Parse the lines of a *.con concept file into entity objects
    """
    con_lines = sample["con"].splitlines()

    text = sample["txt"]
    text_lines = text.splitlines()
    text_line_lengths = [len(el) for el in text_lines]

    # parsed concepts (sort is just a convenience)
    con_parsed = sorted(
        [_parse_con_line(line) for line in con_lines],
        key=lambda x: (x["start_line"], x["start_token"]),
    )

    entities = []
    for ii_cp, cp in enumerate(con_parsed):

        # annotations can span multiple lines
        # we loop over all lines and build up the character offsets
        for ii_line in range(cp["start_line"], cp["end_line"] + 1):

            # character offset to the beginning of the line
            # line length of each line + 1 new line character for each line
            start_line_off = sum(text_line_lengths[: ii_line - 1]) + (ii_line - 1)

            # offsets for each token relative to the beginning of the line
            # "one two" -> [(0,3), (4,6)]
            tokoff = _tokoff_from_line(text_lines[ii_line - 1])

            # if this is a single line annotation
            if ii_line == cp["start_line"] == cp["end_line"]:
                start_off = start_line_off + tokoff[cp["start_token"]][0]
                end_off = start_line_off + tokoff[cp["end_token"]][1]

            # if multi-line and on first line
            # end_off gets a +1 for new line character
            elif (ii_line == cp["start_line"]) and (ii_line != cp["end_line"]):
                start_off = start_line_off + tokoff[cp["start_token"]][0]
                end_off = start_line_off + text_line_lengths[ii_line - 1] + 1

            # if multi-line and on last line
            elif (ii_line != cp["start_line"]) and (ii_line == cp["end_line"]):
                end_off = end_off + tokoff[cp["end_token"]][1]

            # if mult-line and not on first or last line
            # (this does not seem to occur in this corpus)
            else:
                end_off += text_line_lengths[ii_line - 1] + 1

        text_slice = text[start_off:end_off]
        text_slice_norm_1 = text_slice.replace("\n", "").lower()
        text_slice_norm_2 = text_slice.replace("\n", " ").lower()
        match = text_slice_norm_1 == cp["text"] or text_slice_norm_2 == cp["text"]
        if not match:
            continue

        entity_id = _form_entity_id(
            sample_id,
            split,
            cp["start_line"],
            cp["start_token"],
            cp["end_line"],
            cp["end_token"],
        )
        entity = {
            "id": entity_id,
            "offsets": [(start_off, end_off)],
            # this is the difference between taking text from the entity
            # or taking the text from the offsets. the differences are
            # almost all casing with some small number of new line characters
            # making up the rest
            # "text": [cp["text"]],
            "text": [text_slice],
            "type": cp["type"],
            "normalized": [],
        }
        entities.append(entity)

    # IDs are constructed such that duplicate IDs indicate duplicate (i.e. redundant) entities
    # In practive this removes one duplicate sample from the test set
    # {
    #    'id': 'clinical-627-entity-test-122-9-122-9',
    #    'offsets': [(5600, 5603)],
    #    'text': ['her'],
    #    'type': 'person'
    # }
    dedupe_entities = []
    dedupe_entity_ids = set()
    for entity in entities:
        if entity["id"] in dedupe_entity_ids:
            continue
        else:
            dedupe_entity_ids.add(entity["id"])
            dedupe_entities.append(entity)

    return dedupe_entities


@dataclass
class BigBioConfig(datasets.BuilderConfig):
    """BuilderConfig for BigBio."""
    name: str = None
    version: Version = None
    description: str = None
    schema: str = None
    subset_id: str = None


class N2C22010RelationsDataset(datasets.GeneratorBasedBuilder):
    """i2b2 2010 task comprising concept, assertion and relation extraction"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    # You will be able to load the "source" or "bigbio" configurations with
    # ds_source = datasets.load_dataset('my_dataset', name='source')
    # ds_bigbio = datasets.load_dataset('my_dataset', name='bigbio')

    # For local datasets you can make use of the `data_dir` and `data_files` kwargs
    # https://huggingface.co/docs/datasets/add_dataset.html#downloading-data-files-and-organizing-splits
    # ds_source = datasets.load_dataset('my_dataset', name='source', data_dir="/path/to/data/files")
    # ds_bigbio = datasets.load_dataset('my_dataset', name='bigbio', data_dir="/path/to/data/files")

    BUILDER_CONFIGS = [
        BigBioConfig(
            name=SOURCE,
            version=SOURCE_VERSION,
            description=_DATASETNAME + " source schema",
            schema=SOURCE,
            subset_id=_DATASETNAME,
        ),
        BigBioConfig(
            name=BIGBIO_KB,
            version=BIGBIO_VERSION,
            description=_DATASETNAME + " BigBio schema",
            schema=BIGBIO_KB,
            subset_id=_DATASETNAME,
        )
    ]

    DEFAULT_CONFIG_NAME = _DATASETNAME + "_" + SOURCE

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == SOURCE:
            features = datasets.Features(
                {
                    "doc_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "entities": [
                        {
                            "id": datasets.Value("string"),
                            "offsets": [datasets.Value("int64")],
                            "text": datasets.Value("string"),
                            "concept": datasets.Value("string"),
                            "assertion": datasets.Value("string")
                        }
                    ],
                    "relations": [
                        {
                            "id": datasets.Value("string"),
                            "concept1_text": datasets.Value("string"),
                            "concept1_offsets": [datasets.Value("int64")],
                            "concept2_text": datasets.Value("string"),
                            "concept2_offsets": [datasets.Value("int64")],
                            "relation": datasets.Value("string")
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

        if self.config.data_dir is None:
            raise ValueError("This is a local dataset. Please pass the data_dir kwarg to load_dataset.")
        else:
            data_dir = self.config.data_dir

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "data_dir": data_dir,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_dir": data_dir,
                    "split": "test",
                },
            )
        ]

    @staticmethod
    def _get_source_sample(sample_id, sample):
        return {
            "sample_id": sample_id,
            "txt": sample.get("txt", ""),
            "con": sample.get("con", ""),
            "ast": sample.get("ast", ""),
            "rel": sample.get("rel", ""),
            "metadata": {
                "txt_source": sample.get("txt_source", ""),
                "con_source": sample.get("con_source", ""),
                "ast_source": sample.get("ast_source", ""),
                "rel_source": sample.get("rel_source", ""),
            },
        }

    @staticmethod
    def _get_bigbio_sample(sample_id, sample, split):

        passage_text = sample.get("txt", "")
        entities = _get_entities_from_sample(sample_id, sample, split)
        entity_ids = set([entity["id"] for entity in entities])
        relations = _get_relations_from_sample(sample_id, sample, entity_ids, split)
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

    def _generate_examples(self, data_dir, split) -> (int, dict):
        if split == 'train':
            samples = _read_tar_gz(os.path.join(data_dir, "concept_assertion_relation_training_data.tar.gz"))
        elif split == 'test':
            # This file adds con, ast and rel
            samples = _read_tar_gz(os.path.join(data_dir, "reference_standard_for_test_data.tar.gz"))
            # This file adds txt to already existing samples
            samples = _read_tar_gz(os.path.join(data_dir, "test_data.tar.gz"), samples)

        _id = 0
        for sample_id, sample in samples.items():

            if self.config.name == SOURCE:
                yield _id, self._get_source_sample(sample_id, sample)
            elif self.config.name == BIGBIO_KB:
                print(self._get_bigbio_sample(sample_id, sample, split))
                yield _id, self._get_bigbio_sample(sample_id, sample, split)

            _id += 1


# This allows you to run your dataloader with `python [dataset_name].py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    datasets.load_dataset(__file__,
                          name='source',
                          # name='bigbio_kb',
                          data_dir="/Users/ayush.singh/workspace/data/Healthcare/i2b2_2010_Dataset/")
