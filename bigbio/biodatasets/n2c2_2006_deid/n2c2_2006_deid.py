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
A dataset loader for the n2c2 2006 de-identification dataset.

https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/

The dataset consists of two archive files,

* deid_surrogate_train_all_version2.zip
* deid_surrogate_test_all_groundtruth_version2.zip

The individual data files (inside the zip archives) come in just 1 type:

* xml (*.xml files): contains the id and text of the patient records,
and the corresponding tags for each one of the Patient Health information (PHI)
categories: Patients, Doctors, Hospitals, IDs, Dates, Locations, Phone Numbers, and Ages


The files comprising this dataset must be on the users local machine
in a single directory that is passed to `datasets.load_datset` via
the `data_dir` kwarg. This loader script will read the archive files
directly (i.e. the user should not uncompress, untar or unzip any of
the files). For example, if the following directory structure exists
on the users local machine,


n2c2_2006_deid
├── deid_surrogate_train_all_version2.zip
├── deid_surrogate_test_all_groundtruth_version2.zip


Data Access

from https://www.i2b2.org/NLP/DataSets/Main.php

"As always, you must register AND submit a DUA for access. If you previously
accessed the data sets here on i2b2.org, you will need to set a new password
for your account on the Data Portal, but your original DUA will be retained."


"""
import itertools as it
import os
import re
import xml.etree.ElementTree as et
import zipfile
from typing import Dict, List, Tuple

import datasets

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks

_DATASETNAME = "n2c2_2006"

# https://academic.oup.com/jamia/article/14/5/550/720189
_LANGUAGES = [Lang.EN]
_PUBMED = True
_LOCAL = True
_CITATION = """\
@article{,
    author = {
        Uzuner, Özlem and
        Luo, Yuan and
        Szolovits, Peter
    },
    title     = {Evaluating the State-of-the-Art in Automatic De-identification},
    journal   = {Journal of the American Medical Informatics Association},
    volume    = {14},
    number    = {5},
    pages     = {550-563},
    year      = {2007},
    month     = {09},
    url       = {https://doi.org/10.1197/jamia.M2444},
    doi       = {10.1197/jamia.M2444},
    eprint    = {https://academic.oup.com/jamia/article-pdf/14/5/550/2136261/14-5-550.pdf}
}
"""

_DESCRIPTION = """\
The data for the de-identification challenge came from Partners Healthcare and
included solely medical discharge summaries. We prepared the data for the
challengeby annotating and by replacing all authentic PHI with realistic
surrogates.

Given the above definitions, we marked the authentic PHI in the records in two stages.
In the first stage, we used an automatic system.31 In the second stage, we validated
the output of the automatic system manually. Three annotators, including undergraduate
and graduate students and a professor, serially made three manual passes over each record.
They marked and discussed the PHI tags they disagreed on and finalized these tags
after discussion.

The original dataset does not have spans for each entity. The spans are
computed in this loader and the final text that correspond with the
tags is preserved  in the source format
"""

_HOMEPAGE = "https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/"

_LICENSE = "Data User Agreement"

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class N2C22006DeidDataset(datasets.GeneratorBasedBuilder):
    """n2c2 2006 smoking status identification task"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="n2c2_2006_deid_source",
            version=SOURCE_VERSION,
            description="n2c2_2006 deid source schema",
            schema="source",
            subset_id="n2c2_2006_deid",
        ),
        BigBioConfig(
            name="n2c2_2006_deid_bigbio_kb",
            version=BIGBIO_VERSION,
            description="n2c2_2006 Deid BigBio schema",
            schema="bigbio_kb",
            subset_id="n2c2_2006_deid",
        ),
    ]

    DEFAULT_CONFIG_NAME = "n2c2_2006_deid_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "record_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "phi": datasets.Sequence(datasets.Value("string")),
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

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

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
                    "data_dir": data_dir,
                    "corpus_fname": "deid_surrogate_train_all_version2.zip",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_dir": data_dir,
                    "corpus_fname": "deid_surrogate_test_all_groundtruth_version2.zip",
                },
            ),
        ]

    def _generate_examples(self, data_dir: str, corpus_fname: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        fpath = os.path.join(data_dir, corpus_fname)
        # samples = _read_zip(path)
        if self.config.schema == "source":
            for document in self._generate_parsed_documents(fpath):
                yield document["record_id"], document

        elif self.config.schema == "bigbio_kb":
            uid = it.count(0)
            for document in self._generate_parsed_documents(fpath):
                document["id"] = next(uid)
                document["document_id"] = document.pop("record_id")
                entity_list = document.pop("phi")
                full_text = document.pop("text")
                entities_ = []
                for entity in entity_list:
                    entities_.append(
                        {
                            "id": next(uid),
                            "type": entity["type"],
                            "text": entity["text"],
                            "offsets": entity["offsets"],
                            "normalized": entity["normalized"],
                        }
                    )
                document["entities"] = entities_

                document["passages"] = [
                    {
                        "id": next(uid),
                        "type": "full_text",
                        "text": [full_text],
                        "offsets": [[0, len(full_text)]],
                    },
                ]

                # additional fields required that can be empty
                document["relations"] = []
                document["events"] = []
                document["coreferences"] = []
                yield document["document_id"], document

    def _generate_parsed_documents(self, file_path):
        _, filename = os.path.split(file_path)
        zipped = zipfile.ZipFile(file_path, "r")
        file = zipped.read(filename.split(".")[0] + ".xml")

        # There is an issue with the train file. There is a bad tag in line 25722
        if filename == "deid_surrogate_train_all_version2.zip":
            bad_tag = """<PHI TYPE="DATE">25th of July<PHI TYPE="DOCTOR">""".encode(
                "utf-8"
            )
            replacement_tag = """<PHI TYPE="DATE">25th of July</PHI>""".encode("utf-8")
            file = file.replace(bad_tag, replacement_tag, 1)

        root = et.fromstring(file)
        documents = root.findall("./RECORD")
        record_regex = r"<RECORD ID=|</RECORD>"
        text_regex = r"<TEXT>|</TEXT>"
        file_string = str(file)
        record_matches = list(re.finditer(record_regex, file_string))
        n_matches = len(record_matches)
        if len(documents) != n_matches / 2:
            raise ValueError(
                """the records found thourgh regex are not the
                                same as the ones found using xmltree"""
            )

        # counter for the documents in xml
        k = 0
        for i in range(0, n_matches, 2):

            # find beginning and end of a section
            record_start = record_matches[i].span()[1]
            record_end = record_matches[i + 1].span()[0]
            record_section = file_string[record_start:record_end]

            # find only the text section
            text_matches = list(re.finditer(text_regex, record_section))
            if len(text_matches) > 2:
                raise ValueError("It should only be one match for text within a record")
            text_start = text_matches[0].span()[1]
            text_end = text_matches[1].span()[0]

            # remove new line at the beginning and the end
            full_text_with_tags = record_section[text_start:text_end].strip("\\n")
            # Remove special characters
            full_text_with_tags = self._remove_special_characters(full_text_with_tags)

            # find all the PHI tags to process them one by one
            document = documents[k]
            phi_xml_tags = document.findall("./TEXT/PHI")
            k += 1

            entities, clean_text = self._extract_tags_text_spans(
                full_text_with_tags=full_text_with_tags, phi_list=phi_xml_tags
            )

            document_dict = {
                "record_id": document.attrib["ID"],
                "text": clean_text,
                "phi": entities,
            }

            yield document_dict

    def _extract_tags_text_spans(
        self, full_text_with_tags: str, phi_list: List[et.Element]
    ) -> List[Dict]:
        """
        Method to extract all PHI tags from within the XML
        Note: There are entities with the same text but different tags.
        Example "Head" Type Doctor/Patient
        Because of this the method needs to check first for it and then assumes the retrieval order
        by the xml library and get the proper spans for each one
        """

        entities = []
        for phi in phi_list:
            entity_text = phi.text
            entity_type = phi.attrib["TYPE"]
            phi_regex = re.escape(f"""<PHI TYPE="{entity_type}">{entity_text}</PHI>""")
            phi_match = re.search(phi_regex, full_text_with_tags)
            if phi_match is None:
                print(phi_regex)
                raise ValueError(f"PHI tag {phi_regex} not found")

            entity_start = phi_match.span()[0]
            entity_end = entity_start + len(entity_text)

            # Substitute in the original text to eliminate the current tag
            # only replace the first occurrence
            full_text_with_tags = re.sub(
                pattern=phi_regex, repl=entity_text, string=full_text_with_tags, count=1
            )

            # check that the text within the span is the same as the entity text
            if entity_text != full_text_with_tags[entity_start:entity_end]:
                raise ValueError("Entity text does not have the correct span")

            # save the entities
            entities.append(
                {
                    "text": [entity_text],
                    "type": entity_type,
                    "offsets": [[entity_start, entity_end]],
                    "normalized": [],
                }
            )

        # clean the last remaining tag
        clean_text = re.sub(r"\\n\[ report_end \]", "", full_text_with_tags)
        return entities, clean_text

    def _remove_special_characters(self, text: str) -> str:
        result = text.replace("&gt;", ">")
        result = result.replace("&lt;", "<")
        result = result.replace("&quot;", '"')
        result = result.replace("&apos;", "'")
        return result
