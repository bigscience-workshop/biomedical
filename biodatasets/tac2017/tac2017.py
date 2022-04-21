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
from typing import List, Tuple, Dict

#from lxml import etree
import xml.etree.ElementTree as ET

import datasets
from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

# TODO: Add BibTeX citation
_CITATION = """\
@article{,
  author    = {},
  title     = {},
  journal   = {},
  volume    = {},
  year      = {},
  url       = {},
  doi       = {},
  biburl    = {},
  bibsource = {}
}
"""

_DATASETNAME = "tac2017"

_DESCRIPTION = """\
This dataset is designed for extraction of ADRs from prescription drug labels.
"""
_HOMEPAGE = "https://bionlp.nlm.nih.gov/tac2017adversereactions/"

_LICENSE = "None provided."

# TODO: Add links to the urls needed to download your dataset files.
#  For local datasets, this variable can be an empty dictionary.

# For publicly available datasets you will most likely end up passing these URLs to dl_manager in _split_generators.
# In most cases the URLs will be the same for the source and bigbio config.
# However, if you need to access different files for each config you can have multiple entries in this dict.
# This can be an arbitrarily nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "tac2017": "https://bionlp.nlm.nih.gov/tac2017adversereactions/train_xml.tar.gz",
}

_SUPPORTED_TASKS = [
    Tasks.NAMED_ENTITY_DISAMBIGUATION,
    Tasks.NAMED_ENTITY_RECOGNITION,
    Tasks.RELATION_EXTRACTION
]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"

class Tac2017Dataset(datasets.GeneratorBasedBuilder):
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
    #  If dataset contains more than one subset (see examples/bioasq.py) implement for EACH of them.
    #  Each of them should contain:
    #   - name: should be unique for each dataset config eg. bioasq10b_(source|bigbio)_[bigbio_schema_name]
    #   - version: option = (SOURCE_VERSION|BIGBIO_VERSION)
    #   - description: one line description for the dataset
    #   - schema: options = (source|bigbio_[bigbio_schema_name])
    #   - subset_id: subset id is the canonical name for the dataset (eg. bioasq10b)
    #  where [bigbio_schema_name] = (kb, pairs, qa, text, t2t, entailment)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="tac2017_source",
            version=SOURCE_VERSION,
            description="TAC 2017 source schema",
            schema="source",
            subset_id="tac2017",
        ),
        BigBioConfig(
            name="tac2017_bigbio_[bigbio_schema_name]",
            version=BIGBIO_VERSION,
            description="TAC 2017 BigBio schema",
            schema="bigbio_kb",
            subset_id="tac2017",
        ),
    ]

    DEFAULT_CONFIG_NAME = "tac2017_source"

    def _info(self) -> datasets.DatasetInfo:

        # Create the source schema; this schema will keep all keys/information/labels as close to the original dataset as possible.

        # You can arbitrarily nest lists and dictionaries.
        # For iterables, use lists over tuples or `datasets.Sequence`

        if self.config.schema == "source":
            # TODO: Create your source schema here
            features = datasets.Features(
                {
                    "labels" : {
                        "drug": datasets.Value("string"),
                        "text": {
                            "section" : {
                                "name": datasets.Value("string"),
                                "id": datasets.Value("string"),
                                "text": datasets.Value("string"),
                            },
                            "mentions" : {
                                "id": datasets.Value("string"),
                                "type": datasets.Value("string"),
                                "section": datasets.Value("string"),
                                "start": datasets.Value("int32"),
                                "len": datasets.Value("int32"),
                                "str":datasets.Value("string")
                            },
                            "relations": {
                                "id": datasets.Value("string"),
                                "type": datasets.Value("string"),
                                "arg1": datasets.Value("string"),
                                "arg2": datasets.Value("string")
                            },
                            "reactions": {
                                "id": datasets.Value("string"),
                                "str": datasets.Value("string"),
                                "normalization": {
                                    "id": datasets.Value("string"),
                                    "meddra_pt": datasets.Value("string"),
                                    "meddra_pt_id": datasets.Value("string"),
                                    "meddra_llt": datasets.Value("string"),
                                    "meddra_llt_id": datasets.Value("string")
                                },
                            }
                        }
                    }
                }
            )

            # EX: Arbitrary NER type dataset
            # features = datasets.Features(
            #    {
            #        "doc_id": datasets.Value("string"),
            #        "text": datasets.Value("string"),
            #        "entities": [
            #            {
            #                "offsets": [datasets.Value("int64")],
            #                "text": datasets.Value("string"),
            #                "type": datasets.Value("string"),
            #                "entity_id": datasets.Value("string"),
            #            }
            #        ],
            #    }
            # )

        # Choose the appropriate bigbio schema for your task and copy it here. You can find information on the schemas in the CONTRIBUTING guide.

        # In rare cases you may get a dataset that supports multiple tasks requiring multiple schemas. In that case you can define multiple bigbio configs with a bigbio_[bigbio_schema_name] format.

        # For example bigbio_kb, bigbio_t2t
        elif self.config.schema == "bigbio_kb":
            # e.g. features = schemas.kb_features
            # TODO: Choose your big-bio schema here
            raise NotImplementedError()

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration

        # If you need to access the "source" or "bigbio" config choice, that will be in self.config.name

        # LOCAL DATASETS: You do not need the dl_manager; you can ignore this argument. Make sure `gen_kwargs` in the return gets passed the right filepath

        # PUBLIC DATASETS: Assign your data-dir based on the dl_manager.
        train_fpaths = {
            'train_xml/ADCETRIS.xml',
            'train_xml/BEPREVE.xml',
            'train_xml/CLEVIPREX.xml',
            'train_xml/DYSPORT.xml',
            'train_xml/FERRIPROX.xml',
            'train_xml/ILARIS.xml',
            'train_xml/KALYDECO.xml',
            'train_xml/ONFI.xml',
            'train_xml/SAPHRIS.xml',
            'train_xml/TIVICAY.xml',
            'train_xml/VIZAMYL.xml',
            'train_xml/ZYKADIA.xml',
            'train_xml/ADREVIEW.xml',
            'train_xml/BESIVANCE.xml',
            'train_xml/COARTEM.xml',
            'train_xml/EDARBI.xml',
            'train_xml/FIRAZYR.xml',
            'train_xml/IMBRUVICA.xml',
            'train_xml/KYPROLIS.xml',
            'train_xml/OTEZLA.xml',
            'train_xml/SIMPONI.xml',
            'train_xml/TOVIAZ.xml',
            'train_xml/VORAXAZE.xml',
            'train_xml/ZYTIGA.xml',
            'train_xml/AFINITOR.xml',
            'train_xml/BLINCYTO.xml',
            'train_xml/COMETRIQ.xml',
            'train_xml/ELIQUIS.xml',
            'train_xml/FULYZAQ.xml',
            'train_xml/INLYTA.xml',
            'train_xml/LUMIZYME.xml',
            'train_xml/PICATO.xml',
            'train_xml/SIRTURO.xml',
            'train_xml/TREANDA.xml',
            'train_xml/XALKORI.xml',
            'train_xml/AMPYRA.xml',
            'train_xml/BOSULIF.xml',
            'train_xml/DALVANCE.xml',
            'train_xml/ENTEREG.xml',
            'train_xml/GADAVIST.xml',
            'train_xml/INTELENCE.xml',
            'train_xml/MULTAQ.xml',
            'train_xml/POTIGA.xml',
            'train_xml/STENDRA.xml',
            'train_xml/TRULICITY.xml',
            'train_xml/XEOMIN.xml',
            'train_xml/AMYVID.xml',
            'train_xml/BREO.xml',
            'train_xml/DATSCAN.xml',
            'train_xml/EOVIST.xml',
            'train_xml/GILENYA.xml',
            'train_xml/INVOKANA.xml',
            'train_xml/NATAZIA.xml',
            'train_xml/PRADAXA.xml',
            'train_xml/STRIBILD.xml',
            'train_xml/TUDORZA.xml',
            'train_xml/XIAFLEX.xml',
            'train_xml/APTIOM.xml',
            'train_xml/CARBAGLU.xml',
            'train_xml/DIFICID.xml',
            'train_xml/ERWINAZE.xml',
            'train_xml/GILOTRIF.xml',
            'train_xml/JARDIANCE.xml',
            'train_xml/NESINA.xml',
            'train_xml/PRISTIQ.xml',
            'train_xml/TAFINLAR.xml',
            'train_xml/ULESFIA.xml',
            'train_xml/XTANDI.xml',
            'train_xml/ARCAPTA.xml',
            'train_xml/CERDELGA.xml',
            'train_xml/DOTAREM.xml',
            'train_xml/EYLEA.xml',
            'train_xml/GRANIX.xml',
            'train_xml/JEVTANA.xml',
            'train_xml/NEURACEQ.xml',
            'train_xml/PROLIA.xml',
            'train_xml/TANZEUM.xml',
            'train_xml/ULORIC.xml',
            'train_xml/YERVOY.xml',
            'train_xml/BELEODAQ.xml',
            'train_xml/CHOLINE.xml',
            'train_xml/DUAVEE.xml',
            'train_xml/FANAPT.xml',
            'train_xml/HALAVEN.xml',
            'train_xml/JUBLIA.xml',
            'train_xml/NORTHERA.xml',
            'train_xml/PROMACTA.xml',
            'train_xml/TECFIDERA.xml',
            'train_xml/VICTRELIS.xml',
            'train_xml/ZERBAXA.xml',
            'train_xml/BENLYSTA.xml',
            'train_xml/CIMZIA.xml',
            'train_xml/DUREZOL.xml',
            'train_xml/FARXIGA.xml',
            'train_xml/HORIZANT.xml',
            'train_xml/KALBITOR.xml',
            'train_xml/NULOJIX.xml',
            'train_xml/QUTENZA.xml',
            'train_xml/TEFLARO.xml',
            'train_xml/VIMIZIM.xml',
            'train_xml/ZYDELIG.xml'
        }

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLs; many examples use the download_and_extract method; see the DownloadManager docs here: https://huggingface.co/docs/datasets/package_reference/builder_classes.html#datasets.DownloadManager

        # dl_manager can accept any type of nested list/dict and will give back the same structure with the url replaced with the path to local files.

        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs= {
                    "files": [os.path.join(data_dir, path) for path in train_fpaths],
                    "split": "train",
                },
            ),
        ]  
        

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`

    # TODO: change the args of this function to match the keys in `gen_kwargs`. You may add any necessary kwargs.

    def _generate_examples(self, files, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        # TODO: This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.

        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.

        # NOTE: For local datasets you will have access to self.config.data_dir and self.config.data_files
        print(files)

        if self.config.schema == "source":
            for file in files:
                with open(file) as xml_file:
                    tree = ET.parse(xml_file)
                    label = tree.getroot()
                    #print(label.attrib["drug"])
                    uid = 0
                    for child in label:
                        example = {} 
                        if child.tag == "Text":
                            text = child
                            for section in text:
                                example["section"] = section
                                print(section.attrib["id"])
                                print(section.text)
                    #yield uid, example
                    uid +=1

            # # TODO: yield (key, example) tuples in the original dataset schema
            # for key, example in thing:
            #     yield key, example

        elif self.config.schema == "bigbio_kb":
            # TODO: yield (key, example) tuples in the bigbio schema
            for key, example in thing:
                yield key, example


# This template is based on the following template from the datasets package:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py


# This allows you to run your dataloader with `python [dataset_name].py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    datasets.load_dataset(__file__)
