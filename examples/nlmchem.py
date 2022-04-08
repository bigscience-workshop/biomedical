# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
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

import os
import re
from typing import Dict, Iterator, List, Tuple

import bioc
import datasets

from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks
from utils.parsing import get_texts_and_offsets_from_bioc_ann

_CITATION = """\
@Article{islamaj2021nlm,
title={NLM-Chem, a new resource for chemical entity recognition in PubMed full text literature},
author={Islamaj, Rezarta and Leaman, Robert and Kim, Sun and Kwon, Dongseop and Wei, Chih-Hsuan and Comeau, Donald C and Peng, Yifan and Cissel, David and Coss, Cathleen and Fisher, Carol and others},
journal={Scientific Data},
volume={8},
number={1},
pages={1--12},
year={2021},
publisher={Nature Publishing Group}
}
"""

_DESCRIPTION = """\
NLM-Chem corpus consists of 150 full-text articles from the PubMed Central Open Access dataset,
comprising 67 different chemical journals, aiming to cover a general distribution of usage of chemical
names in the biomedical literature.
Articles were selected so that human annotation was most valuable (meaning that they were rich in bio-entities,
and current state-of-the-art named entity recognition systems disagreed on bio-entity recognition.
"""

_HOMEPAGE = "https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vii/track-2"
_LICENSE = " CC0 1.0 Universal (CC0 1.0) Public Domain Dedication"

# files found here `https://ftp.ncbi.nlm.nih.gov/pub/lu/BC7-NLM-Chem-track/` have issues at extraction
# _URLs = {"biocreative": "https://ftp.ncbi.nlm.nih.gov/pub/lu/NLMChem" }
_URLs = {
    "source": "https://ftp.ncbi.nlm.nih.gov/pub/lu/BC7-NLM-Chem-track/BC7T2-NLMChem-corpus_v2.BioC.xml.gz",
    "bigbio_kb": "https://ftp.ncbi.nlm.nih.gov/pub/lu/BC7-NLM-Chem-track/BC7T2-NLMChem-corpus_v2.BioC.xml.gz",
}
_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.NAMED_ENTITY_DISAMBIGUATION]
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class NLMChemDataset(datasets.GeneratorBasedBuilder):
    """NLMChem"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="nlmchem_source",
            version=SOURCE_VERSION,
            description="NLM_Chem source schema",
            schema="source",
            subset_id="nlmchem",
        ),
        BigBioConfig(
            name="nlmchem_bigbio_kb",
            version=BIGBIO_VERSION,
            description="NLM_Chem BigBio schema",
            schema="bigbio_kb",
            subset_id="nlmchem",
        ),
    ]

    DEFAULT_CONFIG_NAME = "nlmchem_source"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):

        if self.config.schema == "source":
            # this is a variation on the BioC format
            features = datasets.Features(
                {
                    "passages": [
                        {
                            "document_id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "text": datasets.Value("string"),
                            "offset": datasets.Value("int32"),
                            "entities": [
                                {
                                    "id": datasets.Value("string"),
                                    "offsets": [[datasets.Value("int32")]],
                                    "text": [datasets.Value("string")],
                                    "type": datasets.Value("string"),
                                    "normalized": [
                                        {
                                            "db_name": datasets.Value("string"),
                                            "db_id": datasets.Value("string"),
                                        }
                                    ],
                                }
                            ],
                        }
                    ]
                }
            )

        elif self.config.schema == "bigbio_kb":
            features = schemas.kb_features

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLs
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        my_urls = _URLs[self.config.schema]
        data_dir = dl_manager.download_and_extract(my_urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, "BC7T2-NLMChem-corpus-train.BioC.xml"
                    ),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, "BC7T2-NLMChem-corpus-test.BioC.xml"
                    ),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, "BC7T2-NLMChem-corpus-dev.BioC.xml"
                    ),
                    "split": "dev",
                },
            ),
        ]

    def _get_passages_and_entities(
        self, d: bioc.BioCDocument
    ) -> Tuple[List[Dict], List[List[Dict]]]:

        passages: List[Dict] = []
        entities: List[List[Dict]] = []

        text_total_length = 0

        po_start = 0

        for _, p in enumerate(d.passages):

            eo = p.offset - text_total_length

            text_total_length += len(p.text) + 1

            po_end = po_start + len(p.text)

            # annotation used only for document indexing
            if p.text is None:
                continue

            dp = {
                "text": p.text,
                "type": p.infons.get("type"),
                "offsets": [(po_start, po_end)],
                "offset": p.offset,  # original offset
            }

            po_start = po_end + 1

            passages.append(dp)

            pe = []

            for a in p.annotations:

                a_type = a.infons.get("type")

                # no in-text annotation: only for document indexing
                if a_type in ["MeSH_Indexing_Chemical", "OTHER"]:
                    continue

                offsets, text = get_texts_and_offsets_from_bioc_ann(a)

                da = {
                    "type": a_type,
                    "offsets": [(start - eo, end - eo) for (start, end) in offsets],
                    "text": text,
                    "id": a.id,
                    "normalized": self._get_normalized(a),
                }

                pe.append(da)

            entities.append(pe)

        return passages, entities

    def _get_normalized(self, a: bioc.BioCAnnotation) -> List[Dict]:
        """
        Get normalization DB and ID from annotation identifiers
        """

        identifiers = a.infons.get("identifier")

        if identifiers is not None:

            identifiers = re.split(r",|;", identifiers)

            identifiers = [i for i in identifiers if i != "-"]

            normalized = [i.split(":") for i in identifiers]

            normalized = [
                {"db_name": elems[0], "db_id": elems[1]} for elems in normalized
            ]

        else:

            normalized = [{"db_name": "-1", "db_id": "-1"}]

        return normalized

    def _generate_examples(
        self,
        filepath: str,
        split: str,  # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    ) -> Iterator[Tuple[int, Dict]]:
        """Yields examples as (key, example) tuples."""

        reader = bioc.BioCXMLDocumentReader(str(filepath))

        if self.config.schema == "source":

            for uid, doc in enumerate(reader):

                passages, passages_entities = self._get_passages_and_entities(doc)

                for p, pe in zip(passages, passages_entities):

                    p.pop("offsets")  # BioC has only start for passages offsets

                    p["document_id"] = doc.id
                    p["entities"] = pe  # BioC has per passage entities

                yield uid, {"passages": passages}

        elif self.config.schema == "bigbio_kb":
            uid = 0
            for idx, doc in enumerate(reader):

                # global id
                uid += 1

                passages, passages_entities = self._get_passages_and_entities(doc)

                # unpack per-passage entities
                entities = [e for pe in passages_entities for e in pe]

                for p in passages:
                    p.pop("offset")  # drop original offset
                    p["text"] = (p["text"],)  # text in passage is Sequence
                    p["id"] = uid  # override BioC default id
                    uid += 1

                for e in entities:
                    e["id"] = uid  # override BioC default id
                    uid += 1

                # if split == "validation" and uid == 6705:
                #     breakpoint()

                yield idx, {
                    "id": uid,
                    "document_id": doc.id,
                    "passages": passages,
                    "entities": entities,
                    "events": [],
                    "coreferences": [],
                    "relations": [],
                }
