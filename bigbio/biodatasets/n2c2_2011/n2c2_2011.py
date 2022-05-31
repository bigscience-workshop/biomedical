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
A dataset loader for the n2c2 2011 coref dataset.

https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/

The dataset consists of four archive files,

* Task_1C.zip
* Task_1C_Test_groundtruth.zip
* i2b2_Partners_Train_Release.tar.gz
* i2b2_Beth_Train_Release.tar.gz

The individual data files (inside the zip and tar archives) come in 4 types,

* docs (*.txt files): text of a patient record
* concepts (*.txt.con files): entities used as input to a coreference model
* chains (*.txt.chains files): chains (i.e. one or more) coreferent entities
* pairs (*.txt.pairs files): pairs of coreferent entities (not required)


The files comprising this dataset must be on the users local machine
in a single directory that is passed to `datasets.load_datset` via
the `data_dir` kwarg. This loader script will read the archive files
directly (i.e. the user should not uncompress, untar or unzip any of
the files). For example, if the following directory structure exists
on the users local machine,


n2c2_2011_coref
├── i2b2_Beth_Train_Release.tar.gz
├── i2b2_Partners_Train_Release.tar.gz
├── Task_1C_Test_groundtruth.zip
└── Task_1C.zip


Data Access

from https://www.i2b2.org/NLP/DataSets/Main.php

"As always, you must register AND submit a DUA for access. If you previously
accessed the data sets here on i2b2.org, you will need to set a new password
for your account on the Data Portal, but your original DUA will be retained."


"""

import os
import re
import tarfile
import zipfile
from collections import defaultdict
from typing import Dict, List, Match, Tuple

import datasets
from datasets import Features, Value

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks

_DATASETNAME = "n2c2_2011"

# https://academic.oup.com/jamia/article/19/5/786/716138
_LANGUAGES = [Lang.EN]
_PUBMED = False
_LOCAL = True
_CITATION = """\
@article{,
    author = {
        Uzuner, Ozlem and
        Bodnari, Andreea and
        Shen, Shuying and
        Forbush, Tyler and
        Pestian, John and
        South, Brett R
    },
    title = "{Evaluating the state of the art in coreference resolution for electronic medical records}",
    journal = {Journal of the American Medical Informatics Association},
    volume = {19},
    number = {5},
    pages = {786-791},
    year = {2012},
    month = {02},
    issn = {1067-5027},
    doi = {10.1136/amiajnl-2011-000784},
    url = {https://doi.org/10.1136/amiajnl-2011-000784},
    eprint = {https://academic.oup.com/jamia/article-pdf/19/5/786/17374287/19-5-786.pdf},
}
"""

_DESCRIPTION = """\
The i2b2/VA corpus contained de-identified discharge summaries from Beth Israel
Deaconess Medical Center, Partners Healthcare, and University of Pittsburgh Medical
Center (UPMC). In addition, UPMC contributed de-identified progress notes to the
i2b2/VA corpus. This dataset contains the records from Beth Israel and Partners.

The i2b2/VA corpus contained five concept categories: problem, person, pronoun,
test, and treatment. Each record in the i2b2/VA corpus was annotated by two
independent annotators for coreference pairs. Then the pairs were post-processed
in order to create coreference chains. These chains were presented to an adjudicator,
who resolved the disagreements between the original annotations, and added or deleted
annotations as necessary. The outputs of the adjudicators were then re-adjudicated, with
particular attention being paid to duplicates and enforcing consistency in the annotations.

"""

_HOMEPAGE = "https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/"

_LICENSE = "Data User Agreement"

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"

_SUPPORTED_TASKS = [Tasks.COREFERENCE_RESOLUTION]


def _read_tar_gz(file_path, samples=None):
    if samples is None:
        samples = defaultdict(dict)
    with tarfile.open(file_path, "r:gz") as tf:
        for member in tf.getmembers():

            base, filename = os.path.split(member.name)
            _, ext = os.path.splitext(filename)
            ext = ext[1:]  # get rid of dot
            sample_id = filename.split(".")[0]

            if ext in ["txt", "con", "pairs", "chains"]:
                samples[sample_id][f"{ext}_source"] = (
                    os.path.basename(file_path) + "|" + member.name
                )
                with tf.extractfile(member) as fp:
                    content_bytes = fp.read()
                content = content_bytes.decode("utf-8")
                samples[sample_id][ext] = content

    return samples


def _read_zip(file_path, samples=None):
    if samples is None:
        samples = defaultdict(dict)
    with zipfile.ZipFile(file_path) as zf:
        for info in zf.infolist():

            base, filename = os.path.split(info.filename)
            _, ext = os.path.splitext(filename)
            ext = ext[1:]  # get rid of dot
            sample_id = filename.split(".")[0]

            if ext in ["txt", "con", "pairs", "chains"] and not filename.startswith(
                "."
            ):
                samples[sample_id][f"{ext}_source"] = (
                    os.path.basename(file_path) + "|" + info.filename
                )
                content = zf.read(info).decode("utf-8")
                samples[sample_id][ext] = content

    return samples


C_PATTERN = r"c=\"(.+?)\" (\d+):(\d+) (\d+):(\d+)"
T_PATTERN = r"t=\"(.+?)\""


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
      'c="<string>" <start_line>:<start_token> <end_line>:<end_token>||t="<type>"'

    """
    c_part, t_part = line.split("||")
    c_match, t_match = re.match(C_PATTERN, c_part), re.match(T_PATTERN, t_part)
    return _ct_match_to_dict(c_match, t_match)


def _parse_chains_line(line: str) -> List[Dict]:
    """Parse one line from a *.chains file.

    A typical line has a chain of concepts and then a type.
      'c="patient" 12:0 12:0||c="mr. andersen" 19:0 19:1||...||t="coref person"'
    """
    pieces = line.split("||")
    c_parts, t_part = pieces[:-1], pieces[-1]
    c_matches, t_match = (
        [re.match(C_PATTERN, c_part) for c_part in c_parts],
        re.match(T_PATTERN, t_part),
    )
    return [_ct_match_to_dict(c_match, t_match) for c_match in c_matches]


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


def _get_corefs_from_sample(sample_id, sample, sample_entity_ids, split):
    """Parse the lines of a *.chains file into coreference objects

    A small number of concepts from the *.con files could not be
    aligned with the text and were excluded. For this reason we
    pass in the full set of matched entity IDs and ensure that
    no coreference refers to an exlcluded entity.
    """
    chains_lines = sample["chains"].splitlines()
    chains_parsed = [_parse_chains_line(line) for line in chains_lines]
    corefs = []
    for ii_cp, cp in enumerate(chains_parsed):
        coref_id = f"{sample_id}-coref-{ii_cp}"
        coref_entity_ids = [
            _form_entity_id(
                sample_id,
                split,
                entity["start_line"],
                entity["start_token"],
                entity["end_line"],
                entity["end_token"],
            )
            for entity in cp
        ]
        coref_entity_ids = [
            ent_id for ent_id in coref_entity_ids if ent_id in sample_entity_ids
        ]
        coref = {
            "id": coref_id,
            "entity_ids": coref_entity_ids,
        }
        corefs.append(coref)

    return corefs


def _get_entities_from_sample(sample_id, sample, split):
    """Parse the lines of a *.con concept file into entity objects

    Here we parse the *.con files and form entities. For a small
    number of entities the text snippet in the concept file could not
    be aligned with the slice from the full text produced by using
    the line and token offsets. These entities are excluded from the
    entities object and the coreferences object.
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


class N2C22011CorefDataset(datasets.GeneratorBasedBuilder):
    """n2c2 2011 coreference task"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="n2c2_2011_source",
            version=SOURCE_VERSION,
            description="n2c2_2011 source schema",
            schema="source",
            subset_id="n2c2_2011",
        ),
        BigBioConfig(
            name="n2c2_2011_bigbio_kb",
            version=BIGBIO_VERSION,
            description="n2c2_2011 BigBio schema",
            schema="bigbio_kb",
            subset_id="n2c2_2011",
        ),
    ]

    DEFAULT_CONFIG_NAME = "n2c2_2011_source"

    def _info(self):

        if self.config.schema == "source":
            features = Features(
                {
                    "sample_id": Value("string"),
                    "txt": Value("string"),
                    "con": Value("string"),
                    "pairs": Value("string"),
                    "chains": Value("string"),
                    "metadata": {
                        "txt_source": Value("string"),
                        "con_source": Value("string"),
                        "pairs_source": Value("string"),
                        "chains_source": Value("string"),
                    },
                }
            )

        elif self.config.schema == "bigbio_kb":
            features = schemas.kb_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:

        if self.config.data_dir is None:
            raise ValueError(
                "This is a local dataset. Please pass the data_dir kwarg to load_dataset."
            )
        else:
            data_dir = self.config.data_dir

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": "train",
                    "data_dir": data_dir,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "split": "test",
                    "data_dir": data_dir,
                },
            ),
        ]

    @staticmethod
    def _get_source_sample(sample_id, sample):
        return {
            "sample_id": sample_id,
            "txt": sample.get("txt", ""),
            "con": sample.get("con", ""),
            "pairs": sample.get("pairs", ""),
            "chains": sample.get("chains", ""),
            "metadata": {
                "txt_source": sample.get("txt_source", ""),
                "con_source": sample.get("con_source", ""),
                "pairs_source": sample.get("pairs_source", ""),
                "chains_source": sample.get("chains_source", ""),
            },
        }

    @staticmethod
    def _get_coref_sample(sample_id, sample, split):

        passage_text = sample.get("txt", "")
        entities = _get_entities_from_sample(sample_id, sample, split)
        entity_ids = set([entity["id"] for entity in entities])
        coreferences = _get_corefs_from_sample(sample_id, sample, entity_ids, split)
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
            "relations": [],
            "events": [],
            "coreferences": coreferences,
        }

    def _generate_examples(self, split, data_dir):
        """Generate samples using the info passed in from _split_generators."""

        if split == "train":
            _id = 0
            # These files have complete sample info
            # (so we get a fresh `samples` defaultdict from each)
            paths = [
                os.path.join(data_dir, "i2b2_Beth_Train_Release.tar.gz"),
                os.path.join(data_dir, "i2b2_Partners_Train_Release.tar.gz"),
            ]
            for path in paths:
                samples = _read_tar_gz(path)
                for sample_id, sample in samples.items():
                    if self.config.schema == "source":
                        yield _id, self._get_source_sample(sample_id, sample)
                    elif self.config.schema == "bigbio_kb":
                        yield _id, self._get_coref_sample(sample_id, sample, split)
                    _id += 1

        elif split == "test":
            _id = 0
            # Information from these files has to be combined to create a full sample
            # (so we pass the `samples` defaultdict back to the `_read_zip` method)
            paths = [
                os.path.join(data_dir, "Task_1C.zip"),
                os.path.join(data_dir, "Task_1C_Test_groundtruth.zip"),
            ]
            samples = defaultdict(dict)
            for path in paths:
                samples = _read_zip(path, samples=samples)

            for sample_id, sample in samples.items():
                if self.config.schema == "source":
                    yield _id, self._get_source_sample(sample_id, sample)
                elif self.config.schema == "bigbio_kb":
                    yield _id, self._get_coref_sample(sample_id, sample, split)
                _id += 1
