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
A dataset loader for the n2c2 2018 Adverse Drug Events and Medication Extraction dataset.

The dataset consists of multiple archive files two of which are being used by the script,
├── training_20180910.zip
└── gold-standard-test-data.zip

The individual data files (inside the zip and tar archives) come in 4 types,

* docs (*.txt files): text of a patient record
* annotations (*.ann files): entities and relations along with offsets used as input to a NER / RE model

The files comprising this dataset must be on the users local machine
in a single directory that is passed to `datasets.load_dataset` via
the `data_dir` kwarg. This loader script will read the archive files
directly (i.e. the user should not uncompress, untar or unzip any of
the files).

Data Access from https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/

[bigbio_schema_name] = kb
"""

import os
import zipfile
from collections import defaultdict
from typing import List, Tuple, Dict

import datasets
from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

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

_DESCRIPTION = """\
The National NLP Clinical Challenges (n2c2), organized in 2018, continued the
legacy of i2b2 (Informatics for Biology and the Bedside), adding 2 new tracks and 2
new sets of data to the shared tasks organized since 2006. Track 2 of 2018
n2c2 shared tasks focused on the extraction of medications, with their signature
information, and adverse drug events (ADEs) from clinical narratives. 
This track built on our previous medication challenge, but added a special focus on ADEs. 

ADEs are injuries resulting from a medical intervention related to a drugs and 
can include allergic reactions, drug interactions, overdoses, and medication errors. 
Collectively, ADEs are estimated to account for 30% of all hospital adverse
events; however, ADEs are preventable. Identifying potential drug interactions,
overdoses, allergies, and errors at the point of care and alerting the caregivers of
potential ADEs can improve health delivery, reduce the risk of ADEs, and improve health
outcomes.  

A step in this direction requires processing narratives of clinical records
that often elaborate on the medications given to a patient, as well as the known
allergies, reactions, and adverse events of the patient. Extraction of this information
from narratives complements the structured medication information that can be
obtained from prescriptions, allowing a more thorough assessment of potential ADEs
before they happen.  

The 2018 n2c2 shared task Track 2, hereon referred to as the ADE track,
tackled these natural language processing tasks in 3 different steps, 
which we refer to as tasks:  
1. Concept Extraction: identification of concepts related to medications, 
their signature information, and ADEs  
2. Relation Classification: linking the previously mentioned concepts to 
their medication  by identifying relations on gold standard concepts  
3. End-to-End: building end-to-end systems that process raw narrative text 
to discover concepts and find relations of those concepts to their medications

Shared tasks provide a venue for head-to-head comparison of systems developed
for the same task and on the same data, allowing researchers to identify the state
of the art in a particular task, learn from it, and build on it.
"""

_HOMEPAGE = "https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/"

_LICENSE = "External Data User Agreement"

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"  # 2018-09-10
_BIGBIO_VERSION = "1.0.0"

# Constants
DELIMITER = "||"
SOURCE = "source"
BIGBIO_KB = "bigbio_kb"
ID = "id"
ANNOTATIONS_EXT = "ann"
TEXT, TEXT_EXT = "text", "txt"
TAG, TAGS = "tag", "tags"
RELATION, RELATIONS = "relation", "relations"
START, END = "start", "end"

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
        ID: tag_id,
        TEXT: tag_text,
        START: int(tag_start),
        END: int(tag_end),
        TAG: tag_type,
    }


def _build_relation_dict(rel_id, arg1, arg2, rel_type):
    return {
        ID: rel_id,
        "arg1_id": arg1,
        "arg2_id": arg2,
        RELATION: rel_type,
    }


def _get_annotations(annotation_file):
    """Return a dictionary with all the annotations in the .ann file.

    A typical line has either of the following form,
        1. 'T41	Form 8977 8990	ophthalmology' -> '<ID> <CONCEPT> <START CHAR OFFSET> <END CHAR OFFSET> <TEXT>'
        2. 'R22	Form-Drug Arg1:T41 Arg2:T40' -> '<ID> <RELATION> <CONCEPT_1_ID> <CONCEPT_2_ID>'

    """
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
        arg1 = tags[rel_arg1][ID]
        arg2 = tags[rel_arg2][ID]
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
                    samples[sample_id][TAGS], samples[sample_id][RELATIONS] = _get_annotations(content)

    return samples


def _get_entities_from_sample(sample_id, sample, split):
    entities = []
    text = sample[TEXT_EXT]
    for entity in sample[TAGS]:
        text_slice = text[entity[START]: entity[END]]
        text_slice_norm_1 = text_slice.replace("\n", "").lower()
        text_slice_norm_2 = text_slice.replace("\n", " ").lower()
        match = text_slice_norm_1 == entity[TEXT] or text_slice_norm_2 == entity[TEXT]
        if not match:
            continue
        entities.append({
            ID: _form_id(sample_id, entity[ID], split,
                         entity[START], entity[END]),
            "type": entity[TAG],
            TEXT: [text_slice],
            "offsets": [(entity[START], entity[END])],
            "normalized": [],
        })

    return entities


def _get_relations_from_sample(sample_id, sample, split):
    relations = []
    for relation in sample[RELATIONS]:
        relations.append({
            ID: _form_id(sample_id, relation[ID], split,
                         relation["arg1_id"], relation["arg2_id"], RELATION),
            "type": relation[RELATION],
            "arg1_id": relation["arg1_id"],
            "arg2_id": relation["arg2_id"],
            "normalized": [],
        })

    return relations


class N2C2AdverseDrugEventsMedicationExtractionDataset(datasets.GeneratorBasedBuilder):
    """n2c2 2018 Track 2 concept and relation task"""

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
                    TEXT: datasets.Value("string"),
                    TAGS: [
                        {
                            ID: datasets.Value("string"),
                            TEXT: datasets.Value("string"),
                            START: datasets.Value("int64"),
                            END: datasets.Value("int64"),
                            TAG: datasets.ClassLabel(names=N2C2_2018_NER_LABELS),
                        }
                    ],
                    RELATIONS: [
                        {
                            ID: datasets.Value("string"),
                            "arg1_id": datasets.Value("string"),
                            "arg2_id": datasets.Value("string"),
                            RELATION: datasets.ClassLabel(names=N2C2_2018_RELATION_LABELS),
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
            TEXT: sample.get(TEXT_EXT, ""),
            TAGS: sample.get(TAGS, []),
            RELATIONS: sample.get(RELATIONS, [])
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
