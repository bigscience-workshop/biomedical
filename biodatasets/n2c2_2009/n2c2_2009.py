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
A dataset loader for the n2c2 2009 medication dataset.

The dataset consists of three archive files,
├── annotations_ground_truth.tar.gz
├── train.test.released.8.17.09.tar.gz
├── TeamSubmissions.zip
└── training.sets.released.tar.gz

The individual data files (inside the zip and tar archives) come in 4 types,

* entries (*.entries / no extension files): text of a patient record
* medications (*.m files): entities along with offsets used as input to a named entity recognition model

The files comprising this dataset must be on the users local machine
in a single directory that is passed to `datasets.load_dataset` via
the `data_dir` kwarg. This loader script will read the archive files
directly (i.e. the user should not uncompress, untar or unzip any of
the files).

Data Access from https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/

Steps taken to build datasets:
1. Read all data files from train.test.released.8.17.09
2. Get IDs of all train files from training.sets.released
3. Intersect 2 with 1 to get train set
4. Difference 1 with 2 to get test set
5. Enrich train set with training.ground.truth.01.06.11.2009
6. Enrich test set with annotations_ground_truth
"""

import os
import re
import tarfile
import zipfile
from collections import defaultdict
from typing import Dict, List, Match, Tuple, Union

# from datasets import Features, Value, DatasetInfo, GeneratorBasedBuilder, Version, SplitGenerator, Split, load_dataset
import datasets
from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

_CITATION = """\
@article{DBLP:journals/jamia/UzunerSC10,
  author    = {
                Ozlem Uzuner and
                Imre Solti and
                Eithon Cadag
               },
  title     = {Extracting medication information from clinical text},
  journal   = {J. Am. Medical Informatics Assoc.},
  volume    = {17},
  number    = {5},
  pages     = {514--518},
  year      = {2010},
  url       = {https://doi.org/10.1136/jamia.2010.003947},
  doi       = {10.1136/jamia.2010.003947},
  timestamp = {Mon, 11 May 2020 22:59:55 +0200},
  biburl    = {https://dblp.org/rec/journals/jamia/UzunerSC10.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_DATASETNAME = "n2c2_2009"

_DESCRIPTION = """\
The Third i2b2 Workshop on Natural Language Processing Challenges for Clinical Records 
focused on the identification of medications, their dosages, modes (routes) of administration, 
frequencies, durations, and reasons for administration in discharge summaries. 
The third i2b2 challenge—that is, the medication challenge—extends information 
extraction to relation extraction; it requires extraction of medications and
medication-related information followed by determination of which medication 
belongs to which medication-related details.

The medication challenge was designed as an information extraction task.
The goal, for each discharge summary, was to extract the following information 
on medications experienced by the patient:
1. Medications (m): including names, brand names, generics, and collective names of prescription substances, 
over the counter medications, and other biological substances for which the patient is the experiencer.
2. Dosages (do): indicating the amount of a medication used in each administration.
3. Modes (mo): indicating the route for administering the medication.
4. Frequencies (f): indicating how often each dose of the medication should be taken.
5. Durations (du): indicating how long the medication is to be administered.
6. Reasons (r): stating the medical reason for which the medication is given.
7. List/narrative (ln): indicating whether the medication information appears in a 
list structure or in narrative running text in the discharge summary.

The medication challenge asked that systems extract the text corresponding to each of the fields 
for each of the mentions of the medications that were experienced by the patients. 

The values for the set of fields related to a medication mention, if presented within a 
two-line window of the mention, were linked in order to create what we defined as an ‘entry’. 
If the value of a field for a mention were not specified within a two-line window, 
then the value ‘nm’ for ‘not mentioned’ was entered and the offsets were left unspecified.

Since the dataset annotations were crowd-sourced, it contains various violations that are handled
throughout the data loader via means of exception catching or conditional statements. e.g.
annotation: anticoagulation, while in text all words are to be separated by space which 
means words at end of sentence will always contain `.` and hence won't be an exact match 
i.e. `anticoagulation` != `anticoagulation.` from doc_id: 818404
"""

_HOMEPAGE = "https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/"

_LICENSE = "External Data User Agreement"

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "1.0.0"  # 18-Aug-2009
_BIGBIO_VERSION = "1.0.0"

DELIMITER = "||"
SOURCE = "source"
BIGBIO_KB = "bigbio_kb"

TEXT_DATA_FIELDNAME = 'txt'
MEDICATIONS_DATA_FIELDNAME = 'med'
OFFSET_PATTERN = r"(.+?)=\"(.+?)\"( .+)?"  # captures -> do="500" 102:6 102:6 and mo="nm"
BINARY_PATTERN = r"(.+?)=\"(.+?)\""
ENTITY_ID = "entity_id"
MEDICATION = "m"
DOSAGE = "do"
MODE_OF_ADMIN = "mo"
FREQUENCY = "f"
DURATION = "du"
REASON = "r"
EVENT = "e"
TEMPORAL = "t"
CERTAINTY = "c"
IS_FOUND_IN_LIST_OR_NARRATIVE = "ln"
NOT_MENTIONED = "nm"

def _read_train_test_data_from_tar_gz(data_dir):
    samples = defaultdict(dict)

    with tarfile.open(os.path.join(data_dir, 'train.test.released.8.17.09.tar.gz'), "r:gz") as tf:
        for member in tf.getmembers():
            if member.name != 'train.test.released.8.17.09':
                _, sample_id = os.path.split(member.name)

                with tf.extractfile(member) as fp:
                    content_bytes = fp.read()
                content = content_bytes.decode("utf-8")
                samples[sample_id][TEXT_DATA_FIELDNAME] = content

    return samples


def _get_train_set(data_dir, train_test_set):
    train_sample_ids = set()

    # Read training set IDs
    with tarfile.open(os.path.join(data_dir, 'training.sets.released.tar.gz'), "r:gz") as tf:
        for member in tf.getmembers():
            if member.name not in list(map(str, range(1, 11))):
                _, sample_id = os.path.split(member.name)
                train_sample_ids.add(sample_id)

    # Extract training set samples using above IDs from combined dataset
    training_set = {}
    for sample_id in train_sample_ids:
        training_set[sample_id] = train_test_set[sample_id]

    return training_set


def _get_test_set(train_set, train_test_set):
    test_set = {}
    for sample_id, sample in train_test_set.items():
        if sample_id not in train_set:
            test_set[sample_id] = sample

    return test_set


def _add_entities_to_train_set(data_dir, train_set):
    with zipfile.ZipFile(os.path.join(data_dir, 'training.ground.truth.01.06.11.2009.zip')) as zf:
        for info in zf.infolist():
            base, filename = os.path.split(info.filename)
            _, ext = os.path.splitext(filename)
            ext = ext[1:]  # get rid of dot

            # Extract sample id from filename pattern `379569_gold.entries`
            sample_id = filename.split(".")[0].split('_')[0]
            if ext == 'entries':
                train_set[sample_id][MEDICATIONS_DATA_FIELDNAME] = zf.read(info).decode("utf-8")


def _add_entities_to_test_set(data_dir, test_set):
    with tarfile.open(os.path.join(data_dir, 'annotations_ground_truth.tar.gz'), "r:gz") as tf:
        for member in tf.getmembers():
            if 'converted.noduplicates.sorted' in member.name:
                base, filename = os.path.split(member.name)
                _, ext = os.path.splitext(filename)
                ext = ext[1:]  # get rid of dot

                sample_id = filename.split(".")[0]
                if ext == 'm':
                    with tf.extractfile(member) as fp:
                        content_bytes = fp.read()
                    test_set[sample_id][MEDICATIONS_DATA_FIELDNAME] = content_bytes.decode("utf-8")


def _make_empty_schema_dict_with_text(text):
    return {
        "text": text,
        "offsets": [{
            "start_line": 0,
            "start_token": 0,
            "end_line": 0,
            "end_token": 0
        }]
    }


def _ct_match_to_dict(c_match: Match) -> dict:
    """Return a dictionary with groups from concept and type regex matches."""
    key = c_match.group(1)
    text = c_match.group(2)
    offsets = c_match.group(3)
    if offsets:
        offsets = offsets.strip()
        offsets_formatted = []
        # Pattern: f="monday-wednesday-friday...before hemodialysis...p.r.n." 15:7 15:7,16:0 16:1,16:5 16:5
        if ',' in offsets:
            line_offsets = offsets.split(",")
            for offset in line_offsets:
                start, end = offset.split(" ")
                start_line, start_token = start.split(":")
                end_line, end_token = end.split(":")
                offsets_formatted.append({
                    "start_line": int(start_line),
                    "start_token": int(start_token),
                    "end_line": int(end_line),
                    "end_token": int(end_token)
                })
        else:
            """Handle another edge annotations.ground.truth>984424 which has discontinuous
            annotation as 23:4 23:4 23:10 23:10 which violates annotation guideline that
            discontinuous spans should be separated by comma -> 23:4 23:4,23:10 23:10 
            """
            offset = offsets.split(" ")
            for i in range(0, len(offset), 2):
                start, end = offset[i: i+2]
                start_line, start_token = start.split(":")
                end_line, end_token = end.split(":")

                offsets_formatted.append({
                    "start_line": int(start_line),
                    "start_token": int(start_token),
                    "end_line": int(end_line),
                    "end_token": int(end_token)
                })

        return {
            "text": text,
            "offsets": offsets_formatted
        }
    elif key in {CERTAINTY, EVENT, TEMPORAL, IS_FOUND_IN_LIST_OR_NARRATIVE}:
        return text
    else:
        return _make_empty_schema_dict_with_text(text)


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
        if (char != " " or char != "\t") and start is None:
            start = ii
        if (char == " " or char == "\t") and start is not None:
            end = ii
            tokoff.append((start, end))
            start = None
    if start is not None:
        end = ii + 1
        tokoff.append((start, end))
    return tokoff


def _parse_line(line: str) -> dict:
    """Parse one line from a *.m file.

    A typical line has the form,
      'm="<string>" <start_line>:<start_token> <end_line>:<end_token>||...||e="<string>"||...'

    This represents one medication.
    It can be interpreted as follows,
        Medication name & offset||dosage & offset||mode & offset||frequency & offset||...
        ...duration & offset||reason & offset||event||temporal marker||certainty||list/narrative

    If there is no information then each field will simply contain "nm" (not mentioned)

    Anomalies:
    1. Files 683679 and 974209 annotations do not have 'c', 'e', 't' keys in them
    2. Some files have discontinuous annotations violating guidelines i.e. using space insead of comma as delimiter
    """
    entity = {
        MEDICATION: _make_empty_schema_dict_with_text(''),
        DOSAGE: _make_empty_schema_dict_with_text(''),
        MODE_OF_ADMIN: _make_empty_schema_dict_with_text(''),
        FREQUENCY: _make_empty_schema_dict_with_text(''),
        DURATION: _make_empty_schema_dict_with_text(''),
        REASON: _make_empty_schema_dict_with_text(''),
        EVENT: '',
        TEMPORAL: '',
        CERTAINTY: '',
        IS_FOUND_IN_LIST_OR_NARRATIVE: ''
    }
    for i, pattern in enumerate(line.split(DELIMITER)):
        # Handle edge case of triple pipe as delimiter in 18563_gold.entries: ...7,16:0 16:1,16:5 16:5||| du="nm"...
        if pattern[0] == '|':
            pattern = pattern[1:]

        pattern = pattern.strip()
        match = re.match(OFFSET_PATTERN, pattern)
        key = match.group(1)
        entity[key] = _ct_match_to_dict(match)

    return entity


def _form_entity_id(sample_id, split, start_line, start_token, end_line, end_token):
    return "{}-entity-{}-{}-{}-{}-{}".format(
        sample_id,
        split,
        start_line,
        start_token,
        end_line,
        end_token,
    )


def _get_entities_from_sample(sample_id, sample, split):
    entities = []
    if MEDICATIONS_DATA_FIELDNAME not in sample:
        return entities

    text = sample[TEXT_DATA_FIELDNAME]
    text_lines = text.splitlines()
    text_line_lengths = [len(el) for el in text_lines]
    med_lines = sample[MEDICATIONS_DATA_FIELDNAME].splitlines()
    # parsed concepts (sort is just a convenience)
    med_parsed = sorted(
        [_parse_line(line) for line in med_lines],
        key=lambda x: (x[MEDICATION]['offsets'][0]["start_line"], x[MEDICATION]['offsets'][0]["start_token"]),
    )

    for ii_cp, cp in enumerate(med_parsed):
        for entity_type in {MEDICATION, DOSAGE, DURATION, REASON, FREQUENCY, MODE_OF_ADMIN}:
            offsets, texts = [], []
            for txt, offset in zip(cp[entity_type]['text'].split('...'), cp[entity_type]['offsets']):
                # annotations can span multiple lines
                # we loop over all lines and build up the character offsets
                for ii_line in range(offset["start_line"], offset["end_line"] + 1):

                    # character offset to the beginning of the line
                    # line length of each line + 1 new line character for each line
                    # need to subtract 1 from offset["start_line"] because line index starts at 1 in dataset
                    start_line_off = sum(text_line_lengths[:ii_line - 1]) + (ii_line - 1)

                    # offsets for each token relative to the beginning of the line
                    # "one two" -> [(0,3), (4,6)]
                    tokoff = _tokoff_from_line(text_lines[ii_line - 1])
                    try:
                        # if this is a single line annotation
                        if ii_line == offset["start_line"] == offset["end_line"]:
                            start_off = start_line_off + tokoff[offset["start_token"]][0]
                            end_off = start_line_off + tokoff[offset["end_token"]][1]

                        # if multi-line and on first line
                        # end_off gets a +1 for new line character
                        elif (ii_line == offset["start_line"]) and (ii_line != offset["end_line"]):
                            start_off = start_line_off + tokoff[offset["start_token"]][0]
                            end_off = start_line_off + text_line_lengths[ii_line - 1] + 1
                            if 'anticoagulation' in txt and sample_id == '818404':
                                print('1.', ii_line, start_off, end_off, text[start_off:end_off], txt, offset)
                                print(tokoff)

                        # if multi-line and on last line
                        elif (ii_line != offset["start_line"]) and (ii_line == offset["end_line"]):
                            end_off += tokoff[offset["end_token"]][1]
                            if 'anticoagulation' in txt and sample_id == '818404':
                                print('2.', ii_line, start_off, end_off, repr(text[start_off:end_off]), txt, offset)
                                print(tokoff)

                        # if mult-line and not on first or last line
                        # (this does not seem to occur in this corpus)
                        else:
                            end_off += text_line_lengths[ii_line - 1] + 1

                    except IndexError as e:
                        """This is to handle an erroneous annotation in files #974209 line 51
                        line is 'the PACU in stable condition. Her pain was well controlled with PCA'
                        whereas the annotation says 'pca analgesia' where 'analgesia' is missing from
                        the end of the line. This results in token not being found in `tokoff` array
                        and raises IndexError
                        
                        similar files:
                         * 5091 - amputation beginning two weeks ago associated with throbbing
                         * 944118 - dysuria , joint pain. Reported small rash on penis for which was taking
                         * 918321 - endarterectomy. The patient was started on enteric coated aspirin
                        """
                        continue

                offsets.append((start_off, end_off))

                text_slice = text[start_off:end_off]
                text_slice_norm_1 = text_slice.replace("\n", "").lower()
                text_slice_norm_2 = text_slice.replace("\n", " ").lower()
                match = text_slice_norm_1 == txt or text_slice_norm_2 == txt.lower()
                if not match:
                    continue

                texts.append(text_slice)

            entity_id = _form_entity_id(
                sample_id,
                split,
                cp[entity_type]['offsets'][0]['start_line'],
                cp[entity_type]['offsets'][0]['start_token'],
                cp[entity_type]['offsets'][-1]['end_line'],
                cp[entity_type]['offsets'][-1]['end_token']
            )
            entity = {
                "id": entity_id,
                "offsets": offsets if texts else [],
                "text": texts,
                "type": entity_type,
                "normalized": [],
            }
            entities.append(entity)

    # IDs are constructed such that duplicate IDs indicate duplicate (i.e. redundant) entities
    dedupe_entities = []
    dedupe_entity_ids = set()
    for entity in entities:
        if entity["id"] in dedupe_entity_ids:
            continue
        else:
            dedupe_entity_ids.add(entity["id"])
            dedupe_entities.append(entity)

    return dedupe_entities


class N2C22009MedicationDataset(datasets.GeneratorBasedBuilder):
    """n2c2 2009 Medications NER task"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)
    SOURCE_CONFIG_NAME = _DATASETNAME + "_" + SOURCE
    BIGBIO_CONFIG_NAME = _DATASETNAME + "_" + BIGBIO_KB

    # You will be able to load the "source" or "bigbio" configurations with
    # ds_source = datasets.load_dataset('my_dataset', name='source')
    # ds_bigbio = datasets.load_dataset('my_dataset', name='bigbio')

    # For local datasets you can make use of the `data_dir` and `data_files` kwargs
    # https://huggingface.co/docs/datasets/add_dataset.html#downloading-data-files-and-organizing-splits
    # ds_source = datasets.load_dataset('my_dataset', name='source', data_dir="/path/to/data/files")
    # ds_bigbio = datasets.load_dataset('my_dataset', name='bigbio', data_dir="/path/to/data/files")

    BUILDER_CONFIGS = [
        BigBioConfig(
            name=SOURCE_CONFIG_NAME,
            version=SOURCE_VERSION,
            description=f"{_DATASETNAME} source schema",
            schema=SOURCE,
            subset_id=_DATASETNAME,
        ),
        BigBioConfig(
            name=BIGBIO_CONFIG_NAME,
            version=BIGBIO_VERSION,
            description=f"{_DATASETNAME} BigBio schema",
            schema=BIGBIO_KB,
            subset_id=_DATASETNAME,
        ),
    ]

    DEFAULT_CONFIG_NAME = SOURCE_CONFIG_NAME

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == SOURCE:
            offset_text_schema = {
                "text": datasets.Value("string"),
                "offsets": [
                    {
                        "start_line": datasets.Value("int64"),
                        "start_token": datasets.Value("int64"),
                        "end_line": datasets.Value("int64"),
                        "end_token": datasets.Value("int64"),
                    }
                ]
            }
            features = datasets.Features(
                {
                    "doc_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "entities": [
                        {
                            MEDICATION: offset_text_schema,
                            DOSAGE: offset_text_schema,
                            MODE_OF_ADMIN: offset_text_schema,
                            FREQUENCY: offset_text_schema,
                            DURATION: offset_text_schema,
                            REASON: offset_text_schema,
                            EVENT: datasets.Value("string"),
                            TEMPORAL: datasets.Value("string"),
                            CERTAINTY: datasets.Value("string"),
                            IS_FOUND_IN_LIST_OR_NARRATIVE: datasets.Value("string")
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

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
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
            )
        ]

    @staticmethod
    def _get_source_sample(sample_id, sample) -> Dict[str, Union[str, List[Dict[str, str]]]]:
        entities = []
        if MEDICATIONS_DATA_FIELDNAME in sample:
            entities = list(map(_parse_line, sample[MEDICATIONS_DATA_FIELDNAME].splitlines()))
        return {
            "doc_id": sample_id,
            "text": sample.get(TEXT_DATA_FIELDNAME, ""),
            "entities": entities
        }

    @staticmethod
    def _get_bigbio_sample(sample_id, sample, split) -> Dict[str, Union[str, List[Dict[str, Union[str, List[Tuple]]]]]]:

        passage_text = sample.get(TEXT_DATA_FIELDNAME, "")
        entities = _get_entities_from_sample(sample_id, sample, split)
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
            "coreferences": [],
        }

    def _generate_examples(self, data_dir, split) -> (int, dict):
        train_test_set = _read_train_test_data_from_tar_gz(data_dir)
        train_set = _get_train_set(data_dir, train_test_set)
        test_set = _get_test_set(train_set, train_test_set)

        if split == 'train':
            _add_entities_to_train_set(data_dir, train_set)
            samples = train_set
        elif split == 'test':
            _add_entities_to_test_set(data_dir, test_set)
            samples = test_set

        _id = 0
        for sample_id, sample in samples.items():

            if self.config.name == N2C22009MedicationDataset.SOURCE_CONFIG_NAME:
                yield _id, self._get_source_sample(sample_id, sample)
            elif self.config.name == N2C22009MedicationDataset.BIGBIO_CONFIG_NAME:
                yield _id, self._get_bigbio_sample(sample_id, sample, split)

            _id += 1
