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
A dataset loader for the n2c2 2012 temporal relation dataset.

https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/

The dataset consists of 1 training archive files and 1 annotated test archive file,

* 2012-07-15.original-annotation.release.tar.gz (complete training dataset)
* 2012-08-23.test-data.groundtruth.tar.gz (annotated, complete test dataset)

The files comprising this dataset must be on the users local machine
in a single directory that is passed to `datasets.load_dataset` via
the `data_dir` kwarg. This loader script will read the archive files
directly (i.e. the user should not uncompress, untar or unzip any of
the files).

NOTE. The following XML files are not well formed and have been excluded from
the dataset: "23.xml","53.xml","143.xml","152.xml","272.xml","382.xml","397.xml","422.xml"
"527.xml""547.xml","627.xml","687.xml","802.xml","807.xml".

Registration AND submission of DUA is required to access the dataset.

[bigbio_schema_name] = kb
"""

import os
import tarfile
from collections import defaultdict, OrderedDict
from unittest import skip
import xmltodict
import json
from typing import List, Tuple, Dict

import datasets
from datasets import Features, Value, Sequence, ClassLabel
from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

_CITATION = """\
@article{,
  author    = {
        Sun, Weiyi and
        Rumshisky, Anna and
        Uzuner, Ozlem},
  title     = {Evaluating temporal relations in clinical text: 2012 i2b2 Challenge},
  journal   = {Journal of the American Medical Informatics Association},
  volume    = {20},
  year      = {5},
  pages     = {806-813}
  year      = {2013}
  month     = {09}
  url       = {https://doi.org/10.1136/amiajnl-2013-001628},
  doi       = {10.1136/amiajnl-2013-001628},
  eprint    = {https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3756273/pdf/amiajnl-2013-001628.pdf}
}
"""

_DATASETNAME = "n2c2_2012"

_DESCRIPTION = """\
This dataset is designed for the 2012 i2b2 temporal relations challenge task.

The text annotated for this challenge comes from de-identified discharge summaries. The goal of
the annotation is to mark up temporal information present in clinical text in order to enable
reasoning and queries over the timeline of clinically relevant events for each patient.

This annotation involves marking up three kinds of information:
1) events,
2) temporal expressions, and
3) temporal relations between events and temporal expressions.

The latter would involve: 
1) anchoring events to available temporal expressions, and
2) identifying temporal relations between events.

The first task is to identify all clinically relevant events and situations, including symptoms,
tests, procedures, and other occurrences. The second task is to identify temporal expressions,
which include all expressions related to time, such as dates, times, frequencies, and durations.
Events and temporal expressions have a number of attributes (such as type of event or calendar
value of the temporal expression) that need to be annotated. The final task is to record the
temporal relations (e.g. before, after, simultaneous, etc.) that hold between different events or
between events and temporal expressions.

"""

_HOMEPAGE = "https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/"

_LICENSE = "External Data User Agreement"

_SUPPORTED_TASKS = [Tasks.EVENT_EXTRACTION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"

def _read_tar_gz_train_(file_path, samples=None):
    if samples is None:
        samples = defaultdict(dict)
    with tarfile.open(file_path, "r:gz") as tf:
        for member in tf.getmembers():

            base, filename = os.path.split(member.name)
            _, ext = os.path.splitext(filename)
            ext = ext[1:]  # get rid of dot
            sample_id = filename.split(".")[0]

            if ext == "xml" and not filename in ["23.xml", "143.xml", "152.xml", "272.xml","382.xml","422.xml","547.xml","807.xml"]: # corrputed XML files
                with tf.extractfile(member) as fp:
                    content_bytes = fp.read()
                content = content_bytes.decode("utf-8").encode()
                values = xmltodict.parse(content)
                samples[sample_id] = values["ClinicalNarrativeTemporalAnnotation"] 

    samples_sorted = OrderedDict(sorted(samples.items(),key=lambda x: int(x[0])))
    samples = samples_sorted
    samples = json.loads(json.dumps(samples))

    return samples

def _read_tar_gz_test_(file_path, samples=None):
    if samples is None:
        samples = defaultdict(dict)
    print(samples)
    with tarfile.open(file_path, "r:gz") as tf:
        for member in tf.getmembers():
            if member.name.startswith("ground_truth/merged_xml"):

                base, filename = os.path.split(member.name)
                _, ext = os.path.splitext(filename)
                ext = ext[1:]  # get rid of dot
                sample_id = filename.split(".")[0]

                if ext == "xml" and not filename in ["53.xml", "397.xml","527.xml","627.xml","687.xml","802.xml"]: #corrupted XML files
                    with tf.extractfile(member) as fp:
                        content_bytes = fp.read()
                    content = content_bytes.decode("utf-8").encode()
                    values = xmltodict.parse(content)
                    samples[sample_id] = values["ClinicalNarrativeTemporalAnnotation"] 

    samples_sorted = OrderedDict(sorted(samples.items(), key=lambda x: int(x[0])))
    samples = samples_sorted
    samples = json.loads(json.dumps(samples))

    return samples

def  _get_events_from_sample(sample_id, sample):
    events = []
    for idx, event in enumerate(sample.get("TAGS","").get("EVENT","")):
        
        evs = {
        "id": event.get("@id",""),
        "type": event.get("@type",""),
        "trigger": {
            "text": [event.get("@text","")],
            "offsets": [(int(event.get("@start","")), int(event.get("@end","")))],
            },
        "arguments": [
            {
            "role": [],
            "ref_id": [],
            },
        ],
        }
        events.append(evs)
    return events

def _get_entities_from_sample(sample_id, sample):
    entities = []
    for idx, timex3 in enumerate(sample.get("TAGS","").get("TIMEX3","")):

        entity = {
        "id": timex3.get("@id",""),
        "type": timex3.get("@type",""),
        "offsets": [(int(timex3.get("@start","")), int(timex3.get("@end","")))],
        "text":  [timex3.get("@text","")],
        "normalized": [],
        }

        entities.append(entity)

    return entities

def _get_relations_from_sample(sample_id, sample):

    relations = []
    for idx, tlink in enumerate(sample.get("TAGS").get("TLINK")):

        rel = {
        "id": tlink.get("@id"),
        "type": tlink.get("@type"),
        "arg1_id": tlink.get("@fromID"),
        "arg2_id": tlink.get("@toID"),
        "normalized": [],
        }

        relations.append(rel)

    return relations

def _get_admission_from_sample(sample_id, sample):

    admission = {}

    # When admission information was missing, an empty placeholder was added with id S0
    if sample.get("TAGS","").get("SECTIME","") == "":
        admission = {
            "id": "S0",
            "type": "ADMISSION",
            "text": [],
            "offsets": [],
            }

    elif len(sample.get("TAGS","").get("SECTIME","")) == 2:
        for idx, sectime in enumerate(sample.get("TAGS","").get("SECTIME","")):
            if sectime.get("@type","") == "ADMISSION":
                admission = {
                    "id": sectime.get("@id",""),
                    "type": sectime.get("@type",""),
                    "text": [sectime.get("@text","")],
                    "offsets": [(int(sectime.get("@start","")), int(sectime.get("@end","")))],
                    }

    else:
        sectime = sample.get("TAGS","").get("SECTIME","")
        if sectime.get("@type","") == "ADMISSION":
            admission = {
                "id": sectime.get("@id",""),
                "type": sectime.get("@type",""),
                "text": [sectime.get("@text","")],
                "offsets": [(int(sectime.get("@start","")), int(sectime.get("@end","")))],
                }

    return admission

def _get_discharge_from_sample(sample_id, sample):

    discharge = {}
    
    # When discharge information was missing, an empty placeholder was added with id S1
    if sample.get("TAGS","").get("SECTIME","") == "":
        discharge = {
            "id": "S1",
            "type": "DISCHARGE",
            "text": [],
            "offsets": [],
            }

    elif len(sample.get("TAGS","").get("SECTIME","")) == 2:
        for idx, sectime in enumerate(sample.get("TAGS","").get("SECTIME","")):
            if sectime.get("@type","") == "DISCHARGE":
                discharge = {
                    "id": sectime.get("@id",""),
                    "type": sectime.get("@type",""),
                    "text": [sectime.get("@text","")],
                    "offsets": [(int(sectime.get("@start","")), int(sectime.get("@end","")))],
                    }
    else:
        sectime = sample.get("TAGS","").get("SECTIME","")
        if sectime.get("@type","") == "DISCHARGE":
            discharge = {
                "id": sectime.get("@id",""),
                "type": sectime.get("@type",""),
                "text": [sectime.get("@text","")],
                "offsets": [(int(sectime.get("@start","")), int(sectime.get("@end","")))],
                }

    return discharge


class N2C22012TempRelDataset(datasets.GeneratorBasedBuilder):
    """n2c2 2012 temporal relations challenge"""

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
            name="n2c2_2012_source",
            version=SOURCE_VERSION,
            description="n2c2_2012 source schema",
            schema="source",
            subset_id="n2c2_2012",
        ),
        BigBioConfig(
            name="n2c2_2012_bigbio_kb",
            version=BIGBIO_VERSION,
            description="n2c2_2012 BigBio schema",
            schema="bigbio_kb",
            subset_id="n2c2_2012",
        ),
    ]

    DEFAULT_CONFIG_NAME = "n2c2_2012_source"

    def _info(self) -> datasets.DatasetInfo:

        # Create the source schema; this schema will keep all keys/information/labels as close to the original dataset as possible.

        # You can arbitrarily nest lists and dictionaries.
        # For iterables, use lists over tuples or `datasets.Sequence`

        if self.config.schema == "source":
            features = Features(
                {
                    "sample_id": Value("string"),
                    "text": Value("string"),
                    "tags":{
                        "EVENT": Sequence({"@id": Value("string"),
                                        "@start": Value("int64"),
                                        "@end": Value("int64"),
                                        "@text": Value("string"),
                                        "@modality": Value("string"),
                                        "@polarity": Value("string"),
                                        "@type": Value("string"),
                                        }),
                        "TIMEX3": Sequence({"@id": Value("string"),
                                        "@start": Value("int64"),
                                        "@end": Value("int64"),
                                        "@text": Value("string"),
                                        "@type": Value("string"),
                                        "@val": Value("string"),
                                        "@mod": Value("string"), 
                                        }),
                        "TLINK": Sequence({"@id": Value("string"),
                                        "@fromID": Value("string"),
                                        "@fromText": Value("string"),
                                        "@toID": Value("string"),
                                        "@toText": Value("string"),
                                        "@type": Value("string"),
                                        }),
                        "SECTIME": Sequence({"@id": Value("string"),
                                        "@start": Value("string"),
                                        "@end": Value("string"),
                                        "@text": Value("string"),
                                        "@type": Value("string"),
                                        "@dvalue": Value("string"),                                        
                                        }),
                                  }
                }
            )

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

        if self.config.data_dir is None:
            raise ValueError("This is a local dataset. Please pass the data_dir kwarg to load_dataset.")
        else:
            data_dir = self.config.data_dir

        # Not all datasets have predefined canonical train/val/test splits.
        # If your dataset has no predefined splits, use datasets.Split.TRAIN for all of the data.

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
            ),
        ]

    @staticmethod
    def _get_source_sample(sample_id, sample):
        if sample.get("TAGS","").get("SECTIME","") == "":
            return {
                "sample_id": sample_id,
                "text": sample.get("TEXT",""),
                "tags":{
                    "EVENT": sample.get("TAGS","").get("EVENT",""),
                    "TIMEX3": sample.get("TAGS","").get("TIMEX3",""),
                    "TLINK": sample.get("TAGS","").get("TLINK",""),
                    "SECTIME": [],
                    }
                }
        else:
            return {
                "sample_id": sample_id,
                "text": sample.get("TEXT",""),
                "tags":{
                    "EVENT": sample.get("TAGS","").get("EVENT",""),
                    "TIMEX3": sample.get("TAGS","").get("TIMEX3",""),
                    "TLINK": sample.get("TAGS","").get("TLINK",""),
                    "SECTIME": sample.get("TAGS","").get("SECTIME",""),
                    }
                }

    @staticmethod
    def _get_bigbio_sample(sample_id, sample):

        passage_text = sample.get("TEXT","")
        events = _get_events_from_sample(sample_id, sample)
        entities = _get_entities_from_sample(sample_id, sample)
        relations = _get_relations_from_sample(sample_id, sample)
        admission = _get_admission_from_sample(sample_id, sample)
        discharge = _get_discharge_from_sample(sample_id, sample)

        return {
            "id": sample_id,
            "document_id": sample_id,
            "passages": [
                {
                    "id": f"{sample_id}-full-passage",
                    "type": "Clinical Narrative Temporal Annotation",
                    "text": [passage_text],
                    "offsets": [(0, len(passage_text))],
                },
                admission,
                discharge,
            ],
            "events": events,
            "entities": entities,
            "relations": relations,
            "coreferences": [],
            }


    def _generate_examples(self, data_dir, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        
        if split == "train":
            _id = 0

            file_path = os.path.join(data_dir, "2012-07-15.original-annotation.release.tar.gz")
            samples = _read_tar_gz_train_(file_path)
            for sample_id, sample in samples.items():
                if self.config.schema == "source":
                    yield _id, self._get_source_sample(sample_id, sample)
                elif self.config.schema == "bigbio_kb":
                    yield _id, self._get_bigbio_sample(sample_id, sample)
                _id += 1

        elif split == "test":
            _id = 0

            file_path = os.path.join(data_dir, "2012-08-23.test-data.groundtruth.tar.gz")
            samples = _read_tar_gz_test_(file_path)
            for sample_id, sample in samples.items():
                if self.config.schema == "source":
                    yield _id, self._get_source_sample(sample_id, sample)
                elif self.config.schema == "bigbio_kb":
                    yield _id, self._get_bigbio_sample(sample_id, sample)
                _id += 1
                
# This template is based on the following template from the datasets package:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py