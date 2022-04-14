# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and
#
# TODO: fill out the line below
# * <append your name and optionally your github handle here>
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

"""

import os
import sys
sys.path.append("/Users/patrickhaller/Projects/biomedical")
from typing import List

import bioc
import datasets

from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

# TODO: Add BibTeX citation
_CITATION = """\
@article{,
  author    = {Kevin Nagel},
  title     = {Automatic functional annotation of predicted active sites: combining PDB and literature mining},
  journal   = {},
  volume    = {},
  year      = {2009},
  url       = {https://www.ebi.ac.uk/sites/ebi.ac.uk/files/shared/documents/phdtheses/kevin_nagel.pdf},
  doi       = {},
  biburl    = {},
  bibsource = {}
}
"""

# TODO: create a module level variable with your dataset name (should match script name)
_DATASETNAME = "nagel"

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
This dataset is designed for XXX NLP task.
"""

# TODO: Add a link to an official homepage for the dataset here (if possible)
_HOMEPAGE = "https://sourceforge.net/projects/bionlp-corpora/files/ProteinResidue/"

# Note that this doesn't have to be a common open source license.
# Some datasets have custom licenses. In this case, simply put the full license terms
# into `_LICENSE`
_LICENSE = """
Copyright (c) 2011 Kevin Nagel
Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
"""

# For publicly available datasets you will most likely end up passing these URLs to dl_manager in _split_generators.
# In most cases the URLs will be the same for the source and bigbio config.
# However, if you need to access different files for each config you can have multiple entries in this dict.
# This can be an arbitrarily nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    _DATASETNAME: "https://sourceforge.net/projects/bionlp-corpora/files/ProteinResidue/NagelCorpus.tar.gz/download",
}

# TODO: add supported task by dataset. One dataset may support multiple tasks
_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]

# TODO: set this to a version that is associated with the dataset. if none exists use "1.0.0"
# this version doesn't have to be consistent with semantic versioning. Anything that is
# provided by the original dataset as a version goes.
_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"




# TODO: Name the dataset class to match the script name using CamelCase instead of snake_case
class NagelDataset(datasets.GeneratorBasedBuilder):
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
    #  if dataset contains more than one subset (see examples/bioasq.py) implement for EACH of them. Each of them should contain:
    #  name: should be unique for each dataset config eg. bioasq10b_(source|bigbio)_[bigbioschema_name]
    #  version: option = (SOURCE_VERSION |BIGBIO_VERSION)
    #  description: one line description for the dataset
    #  schema: options = (source|bigbio_[schema_name]) [schema_name] =(kb,pairs, qa, text, test_to_text, entailment)
    #  subset_id: subset id is the canonical name for the dataset (eg. bioasq10b)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="nagel_source",
            version=SOURCE_VERSION,
            description="nagel source schema",
            schema="source",
            subset_id="nagel",
        ),
        BigBioConfig(
            name="nagel_bigbio_kb",
            version=BIGBIO_VERSION,
            description="Nagel BigBio schema",
            schema="bigbio_kb",
            subset_id="nagel",
        )
    ]

    DEFAULT_CONFIG_NAME = "nagel_source"

    def _info(self) -> datasets.DatasetInfo:

        # Create the source schema; this schema will keep all keys/information/labels as close to the original dataset as possible.

        # You can arbitrarily nest lists and dictionaries.
        # For iterables, use lists over tuples or `datasets.Sequence`

        if self.config.schema == "source":
            # TODO: Create your source schema here
            features = datasets.Features(
               {
                   "document_id": datasets.Value("string"),
                   "text": datasets.Value("string"),
                   "entities": [
                       {
                           "offsets": [datasets.Value("int64")],
                           "text": datasets.Value("string"),
                           "type": datasets.Value("string"),
                           "entity_id": datasets.Value("string"),
                       }
                   ],
               }
            )

        # Choose the appropriate bigbio schema for your task and copy it here. You can find information on the schemas in the CONTRIBUTING guide.

        # In rare cases you may get a dataset that supports multiple tasks requiring multiple schemas. In that case you can define multiple bigbio configs with a bigbio_[bigbio_schema_name] format.

        # For example bigbio_kb, bigbio_t2t
        elif self.config.schema =="bigbio_kb":
            features = schemas.kb_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:

        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        # Not all datasets have predefined canonical train/val/test splits. If your dataset does not have any splits, you can omit any missing splits.
        # If your dataset has no predefined splits, use datasets.Split.TRAIN for all of the data.
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "NagelCorpus", "Nagel_GC.xml"),
                    "filepath_standoff": os.path.join(data_dir, "NagelCorpus", "Nagel_GC.standoff.txt"),
                    "folder_path": os.path.join(data_dir, "NagelCorpus", "NagelCorpusText"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "NagelCorpus", "Nagel_GC.xml"),
                    "filepath_standoff": os.path.join(data_dir, "NagelCorpus", "Nagel_GC.xml"),
                    "folder_path": os.path.join(data_dir, "NagelCorpus", "NagelCorpusText"),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "NagelCorpus", "Nagel_GC.xml"),
                    "filepath_standoff": os.path.join(data_dir, "NagelCorpus", "Nagel_GC.xml"),
                    "folder_path": os.path.join(data_dir, "NagelCorpus", "NagelCorpusText"),
                    "split": "dev",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`

    def _generate_examples(self, filepath, filepath_standoff, folder_path, split) -> (int, dict):
       
        from pprint import pprint
        from pathlib import Path

        from utils.parsing import parse_brat_file

        import pandas as pd
        
        print(filepath)
        so_annot = pd.read_csv(
                filepath_standoff,
                delimiter="\t",
                names=[
                    "PMID",
                    "AnnotationType",
                    "Span_start",
                    "Span_End",
                    "AminoAcid_WildType_3_Letter_Abbrev",
                    "ResiduePosition",
                    "AminoAcid_MutatedType_3_Letter_Abbrev",
                    "Residue_mention_in_original_text"
                ]
        )
        
        import xml.etree.ElementTree as ET
            
        xml_abstracts = []

        # XML annotated abstract, but not a read correct xml tree
        # Abstract are divided by a extra newline
        with open(filepath) as f:
            # current_abstract = []
            # xml_lines = f.readlines()
            #
            # for i, line in enumerate(xml_lines):
            #     if line == "\n":
            #         xml_abstracts.append(current_abstract)
            #         current_abstract = []
            #     elif i == len(xml_lines):
            #         xml_abstracts.append(current_abstract)
            #     else:
            #         current_abstract.append(line)
        
            xml_abstracts = f.read().split("\n\n")
            
        # xml_annotations = {}
        #
        # for abstract in xml_abstracts:
        #     # For convenience construct correct xml structure
        #     xml_abstract = ET.fromstring("<root>" + abstract + "</root>")
        #     for child in xml_abstract:
        #         print(child.tag, child.attrib)
        #     exit()
         
        for file in Path(folder_path).iterdir():

            if file.suffix != ".txt":
                continue

            with open(file) as f:
                text = f.read()

            pmid = int(file.name.removesuffix(".txt"))
                
            # Can PMID corresponding annotations
            doc_annotations = so_annot[so_annot.PMID == pmid]

            entities = []
            for annot in doc_annotations.to_dict("records"):

                entities.append({
                    "offsets": [annot["Span_start"], annot["Span_End"]],
                    "type": annot["AnnotationType"],
                    "AminoAcid_WildType_3_Letter_Abbrev": annot["AminoAcid_WildType_3_Letter_Abbrev"],
                    "Residue_Position": annot["Residue_Position"],
                    "AminoAcid_MutatedType_3_Letter_Abbrev": annot["AminoAcid_MutatedType_3_Letter_Abbrev"], "Residue_mention_in_original_text": annot["Residue_mention_in_original_text"]

                })
            yield pmid, {
                "id": pmid,
                "document_id": pmid,
                "text": text,
                "entities": entities
            }



            
        # with open(str(filepath)) as fp:
        #     reader = bioc.biocxml.load(fp)
        #
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.

        # NOTE: For local datasets you will have access to self.config.data_dir and self.config.data_files

        if self.config.schema == "source":
            for uid, doc in enumerate(reader):
                print(uid, doc)

        elif self.config.schema == "bigbio":
            # TODO: yield (key, example) tuples in the bigbio schema
            for key, example in thing:
                yield key, example


# This template is based on the following template from the datasets package:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py


# This allows you to run your dataloader with `python [dataset_name].py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    datasets.load_dataset(__file__)
