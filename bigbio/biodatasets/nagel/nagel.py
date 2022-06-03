# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and
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


import os
from pathlib import Path
from typing import List

import datasets
import pandas as pd

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks
from bigbio.utils.license import CustomLicense

_LANGUAGES = [Lang.EN]
_PUBMED = True
_LOCAL = False
_CITATION = """\
@phdthesis{nagel2009automatic,
  title={Automatic functional annotation of predicted active sites: Combining PDB and literature mining},
  author={Nagel, Kevin},
  year={2009},
  school={University of Cambridge}
  url= {https://www.ebi.ac.uk/sites/ebi.ac.uk/files/shared/documents/phdtheses/kevin_nagel.pdf},
}
"""

_DATASETNAME = "nagel"

_DESCRIPTION = """\
A set of 100 abstracts annotated by Kevin Nagel with protein, residue, organism triples.
"""

_HOMEPAGE = "https://sourceforge.net/projects/bionlp-corpora/files/ProteinResidue/"

_LICENSE = CustomLicense(
    text="""
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
)

_URLS = {
    _DATASETNAME: "https://sourceforge.net/projects/bionlp-corpora/files/ProteinResidue/NagelCorpus.tar.gz/download",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]  # , Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class NagelDataset(datasets.GeneratorBasedBuilder):

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

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
        ),
    ]

    DEFAULT_CONFIG_NAME = "nagel_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "xml_annotated_text": datasets.Value("string"),
                    "entities": [
                        {
                            "offsets": [datasets.Value("int64")],
                            "type": datasets.Value("string"),
                            "AminoAcid_WildType_3_Letter_Abbrev": datasets.Value(
                                "string"
                            ),
                            "Residue_Position": datasets.Value("string"),
                            "AminoAcid_MutatedType_3_Letter_Abbrev": datasets.Value(
                                "string"
                            ),
                            "Residue_mention_in_original_text": datasets.Value(
                                "string"
                            ),
                        }
                    ],
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

        url = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(url)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "NagelCorpus", "Nagel_GC.xml"),
                    "filepath_standoff": os.path.join(
                        data_dir, "NagelCorpus", "Nagel_GC.standoff.txt"
                    ),
                    "folder_path": os.path.join(
                        data_dir, "NagelCorpus", "NagelCorpusText"
                    ),
                    "split": "train",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`

    def _generate_examples(
        self, filepath, filepath_standoff, folder_path, split
    ) -> (int, dict):

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
                "Residue_mention_in_original_text",
            ],
        )

        import xml.etree.ElementTree as ET

        # XML annotated abstract, but not a read correct xml tree
        # Abstract are divided by a extra newline
        with open(filepath) as f:

            # xml_abstracts = f.read().split("\n\n")
            xml_abstracts = "<root>" + f.read() + "</root>"

            root = ET.fromstring(xml_abstracts)

            current_abstract = []
            xml_annotations = {}

            for child in root:

                if child.tag == "rel":
                    current_abstract.append(
                        str(ET.tostring(child, encoding="utf-8", method="xml"))
                    )

                if child.tag == "abs":
                    current_abstract.append(
                        str(ET.tostring(child, encoding="utf-8", method="xml"))
                    )
                    xml_annotations[int(child.attrib["id"])] = "\n".join(
                        current_abstract
                    )
                    current_abstract = []

        if self.config.schema == "source":
            for file in Path(folder_path).iterdir():

                if file.suffix != ".txt":
                    continue

                with open(file) as f:
                    text = f.read()

                pmid = int(file.name.split(".txt")[0])

                # Can PMID corresponding annotations
                doc_annotations = so_annot[so_annot.PMID == pmid]

                entities = []
                for annot in doc_annotations.to_dict("records"):

                    entities.append(
                        {
                            "offsets": [annot["Span_start"], annot["Span_End"]],
                            "type": annot["AnnotationType"],
                            "AminoAcid_WildType_3_Letter_Abbrev": annot[
                                "AminoAcid_WildType_3_Letter_Abbrev"
                            ],
                            "Residue_Position": annot["ResiduePosition"],
                            "AminoAcid_MutatedType_3_Letter_Abbrev": annot[
                                "AminoAcid_MutatedType_3_Letter_Abbrev"
                            ],
                            "Residue_mention_in_original_text": annot[
                                "Residue_mention_in_original_text"
                            ],
                        }
                    )
                yield pmid, {
                    "id": pmid,
                    "document_id": pmid,
                    "text": text,
                    "entities": entities,
                    "xml_annotated_text": xml_annotations[pmid],
                }

        elif self.config.schema == "bigbio_kb":

            entity_id = 0
            passage_id = 0

            for file in Path(folder_path).iterdir():

                if file.suffix != ".txt":
                    continue

                with open(file) as f:
                    text = f.read()

                pmid = int(file.name.split(".txt")[0])

                # Can PMID corresponding annotations
                doc_annotations = so_annot[so_annot.PMID == pmid]

                entities = []

                for annot in doc_annotations.to_dict("records"):

                    # Convert word span to char span
                    span_start = len(" ".join(text.split(" ")[: annot["Span_start"]]))
                    span_end = len(" ".join(text.split(" ")[: annot["Span_End"]]))

                    entities.append(
                        {
                            "id": entity_id,
                            "offsets": [(span_start, span_end)],
                            "type": annot["AnnotationType"],
                            "text": [text[span_start:span_end]],
                            "normalized": [],
                        }
                    )

                    entity_id += 1

                passage_id += 1

                yield str(pmid), {
                    "id": str(pmid),
                    "document_id": str(pmid),
                    "entities": entities,
                    "passages": [
                        {
                            "id": str(passage_id) + "_text",
                            "offsets": [[0, len(text)]],
                            "type": "abstract",
                            "text": [text],
                        }
                    ],
                    "events": [],
                    "coreferences": [],
                    "relations": [],
                }
