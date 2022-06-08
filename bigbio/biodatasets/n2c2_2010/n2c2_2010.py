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
A dataset loader for the n2c2 2010 relations dataset.

The dataset consists of three archive files,
├── concept_assertion_relation_training_data.tar.gz
├── reference_standard_for_test_data.tar.gz
└── test_data.tar.gz

The individual data files (inside the zip and tar archives) come in 4 types,

* docs (*.txt files): text of a patient record
* concepts (*.con files): entities along with offsets used as input to a named entity recognition model
* assertions (*.ast files): entities, offsets and their assertion used as input to a named entity recognition model
* relations (*.rel files): pairs of entities related by relation type used as input to a relation extraction model


The files comprising this dataset must be on the users local machine
in a single directory that is passed to `datasets.load_dataset` via
the `data_dir` kwarg. This loader script will read the archive files
directly (i.e. the user should not uncompress, untar or unzip any of
the files).

Data Access from https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/
"""

import os
import re
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple

import datasets
from datasets import Version

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tags, Tasks
from bigbio.utils.license import Licenses

_TAGS = [Tags.DISEASE, Tags.DIAGNOSIS, Tags.NEGATION]
_LANGUAGES = [Lang.EN]
_PUBMED = False
_LOCAL = True
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

_LICENSE = Licenses.DUA

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
                samples[sample_id][f"{ext}_source"] = (
                    os.path.basename(file_path) + "|" + member.name
                )

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
        "concept": t_match.group(1),
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
    c1_match, r_match, c2_match = (
        re.match(C_PATTERN, c1_part),
        re.match(R_PATTERN, r_part),
        re.match(C_PATTERN, c2_part),
    )
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
    c_match, t_match, a_match = (
        re.match(C_PATTERN, c_part),
        re.match(T_PATTERN, t_part),
        re.match(A_PATTERN, a_part),
    )
    return {
        "text": c_match.group(1),
        "start_line": int(c_match.group(2)),
        "start_token": int(c_match.group(3)),
        "end_line": int(c_match.group(4)),
        "end_token": int(c_match.group(5)),
        "concept": t_match.group(1),
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
        a = {}
        rel = _parse_rel_line(rel_line)
        a["arg1_id"] = _form_entity_id(
            sample_id,
            split,
            rel["concept_1"]["start_line"],
            rel["concept_1"]["start_token"],
            rel["concept_1"]["end_line"],
            rel["concept_1"]["end_token"],
        )
        a["arg2_id"] = _form_entity_id(
            sample_id,
            split,
            rel["concept_2"]["start_line"],
            rel["concept_2"]["start_token"],
            rel["concept_2"]["end_line"],
            rel["concept_2"]["end_token"],
        )
        a["id"] = (
            sample_id + "_" + a["arg1_id"] + "_" + rel["relation"] + "_" + a["arg2_id"]
        )
        a["normalized"] = []
        a["type"] = rel["relation"]
        relations.append(a)

    return relations


def _get_entities_from_sample(sample_id, sample, split):
    """Parse the lines of a *.con concept file into entity objects"""
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
            "type": cp["concept"],
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

    _SOURCE_CONFIG_NAME = _DATASETNAME + "_" + SOURCE
    _BIGBIO_CONFIG_NAME = _DATASETNAME + "_" + BIGBIO_KB

    BUILDER_CONFIGS = [
        BigBioConfig(
            name=_SOURCE_CONFIG_NAME,
            version=SOURCE_VERSION,
            description=_DATASETNAME + " source schema",
            schema=SOURCE,
            subset_id=_DATASETNAME,
        ),
        BigBioConfig(
            name=_BIGBIO_CONFIG_NAME,
            version=BIGBIO_VERSION,
            description=_DATASETNAME + " BigBio schema",
            schema=BIGBIO_KB,
            subset_id=_DATASETNAME,
        ),
    ]

    DEFAULT_CONFIG_NAME = _SOURCE_CONFIG_NAME

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == SOURCE:
            features = datasets.Features(
                {
                    "doc_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "concepts": [
                        {
                            "start_line": datasets.Value("int64"),
                            "start_token": datasets.Value("int64"),
                            "end_line": datasets.Value("int64"),
                            "end_token": datasets.Value("int64"),
                            "text": datasets.Value("string"),
                            "concept": datasets.Value("string"),
                        }
                    ],
                    "assertions": [
                        {
                            "start_line": datasets.Value("int64"),
                            "start_token": datasets.Value("int64"),
                            "end_line": datasets.Value("int64"),
                            "end_token": datasets.Value("int64"),
                            "text": datasets.Value("string"),
                            "concept": datasets.Value("string"),
                            "assertion": datasets.Value("string"),
                        }
                    ],
                    "relations": [
                        {
                            "concept_1": {
                                "text": datasets.Value("string"),
                                "start_line": datasets.Value("int64"),
                                "start_token": datasets.Value("int64"),
                                "end_line": datasets.Value("int64"),
                                "end_token": datasets.Value("int64"),
                            },
                            "concept_2": {
                                "text": datasets.Value("string"),
                                "start_line": datasets.Value("int64"),
                                "start_token": datasets.Value("int64"),
                                "end_line": datasets.Value("int64"),
                                "end_token": datasets.Value("int64"),
                            },
                            "relation": datasets.Value("string"),
                        }
                    ],
                    "unannotated": [
                        {
                            "text": datasets.Value("string"),
                        }
                    ],
                    "metadata": {
                        "txt_source": datasets.Value("string"),
                        "con_source": datasets.Value("string"),
                        "ast_source": datasets.Value("string"),
                        "rel_source": datasets.Value("string"),
                        "unannotated_source": datasets.Value("string"),
                    },
                }
            )

        elif self.config.schema == BIGBIO_KB:
            features = schemas.kb_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:

        if self.config.data_dir is None or self.config.name is None:
            raise ValueError(
                "This is a local dataset. Please pass the data_dir and name kwarg to load_dataset."
            )
        else:
            data_dir = self.config.data_dir

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "data_dir": data_dir,
                    "split": str(datasets.Split.TRAIN),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_dir": data_dir,
                    "split": str(datasets.Split.TEST),
                },
            ),
        ]

    @staticmethod
    def _get_source_sample(sample_id, sample):
        return {
            "doc_id": sample_id,
            "text": sample.get("txt", ""),
            "concepts": list(map(_parse_con_line, sample.get("con", "").splitlines())),
            "assertions": list(
                map(_parse_ast_line, sample.get("ast", "").splitlines())
            ),
            "relations": list(map(_parse_rel_line, sample.get("rel", "").splitlines())),
            "unannotated": sample.get("unannotated", ""),
            "metadata": {
                "txt_source": sample.get("txt_source", ""),
                "con_source": sample.get("con_source", ""),
                "ast_source": sample.get("ast_source", ""),
                "rel_source": sample.get("rel_source", ""),
                "unannotated_source": sample.get("unannotated_source", ""),
            },
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

    def _generate_examples(self, data_dir, split) -> (int, dict):
        if split == "train":
            samples = _read_tar_gz(
                os.path.join(
                    data_dir, "concept_assertion_relation_training_data.tar.gz"
                )
            )
        elif split == "test":
            # This file adds con, ast and rel
            samples = _read_tar_gz(
                os.path.join(data_dir, "reference_standard_for_test_data.tar.gz")
            )
            # This file adds txt to already existing samples
            samples = _read_tar_gz(os.path.join(data_dir, "test_data.tar.gz"), samples)

        _id = 0

        for sample_id, sample in samples.items():

            if self.config.name == N2C22010RelationsDataset._SOURCE_CONFIG_NAME:
                yield _id, self._get_source_sample(sample_id, sample)
            elif self.config.name == N2C22010RelationsDataset._BIGBIO_CONFIG_NAME:
                # This is to make sure unannotated data does not end up in big bio
                if "unannotated" not in sample["txt_source"]:
                    yield _id, self._get_bigbio_sample(sample_id, sample, split)

            _id += 1
