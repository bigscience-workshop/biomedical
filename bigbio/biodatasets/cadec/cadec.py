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
The CADEC corpus (CSIRO Adverse Drug Event Corpus) is is a new rich annotated corpus of medical forum posts on patient-
reported Adverse Drug Events (ADEs). The corpus is sourced from posts on social media, and contains text that is
largely written in colloquial language and often deviates from formal English grammar and punctuation rules.
Annotations contain mentions of concepts such as drugs, adverse events, symptoms, and diseases linked to their
corresponding concepts in controlled vocabularies, i.e., SNOMED Clinical Terms and MedDRA. The quality of the
annotations is ensured by annotation guidelines, multi-stage annotations, measuring inter-annotator agreement, and
final review of the annotations by a clinical terminologist. This corpus is useful for those studies in the area of
information extraction, or more generally text mining, from social media to detect possible adverse drug reactions from
direct patient reports. The dataset contains three views: original (entities annotated in the posts), meddra (entities
normalized with meddra codes), snomedct (entities normalized with SNOMED CT codes).
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from bigbio.utils import parsing, schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks

_LANGUAGES = [Lang.EN]
_LOCAL = False
_CITATION = """\
@article{,
  title={Cadec: A corpus of adverse drug event annotations},
  author={Karimi, Sarvnaz and Metke-Jimenez, Alejandro and Kemp, Madonna and Wang, Chen},
  journal={Journal of biomedical informatics},
  volume={55},
  pages={73--81},
  year={2015},
  publisher={Elsevier}
}
"""

_DATASETNAME = "cadec"

_DESCRIPTION = """\
The CADEC corpus (CSIRO Adverse Drug Event Corpus) is is a new rich annotated corpus of medical forum posts on patient-
reported Adverse Drug Events (ADEs). The corpus is sourced from posts on social media, and contains text that is
largely written in colloquial language and often deviates from formal English grammar and punctuation rules.
Annotations contain mentions of concepts such as drugs, adverse events, symptoms, and diseases linked to their
corresponding concepts in controlled vocabularies, i.e., SNOMED Clinical Terms and MedDRA. The quality of the
annotations is ensured by annotation guidelines, multi-stage annotations, measuring inter-annotator agreement, and
final review of the annotations by a clinical terminologist. This corpus is useful for those studies in the area of
information extraction, or more generally text mining, from social media to detect possible adverse drug reactions from
direct patient reports. The dataset contains three views: original (entities annotated in the posts), meddra (entities
normalized with meddra codes), sct (entities normalized with SNOMED CT codes).
"""

_HOMEPAGE = "https://data.gov.au/dataset/ds-dap-csiro%3A10948/details?q="

_LICENSE = "https://confluence.csiro.au/display/dap/CSIRO+Data+Licence"

_URLS = {
    _DATASETNAME: "https://data.csiro.au/dap/ws/v2/collections/17190/data/1904643",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.NAMED_ENTITY_DISAMBIGUATION]

_SOURCE_VERSION = "2.0.0"

_BIGBIO_VERSION = "1.0.0"


class CadecDataset(datasets.GeneratorBasedBuilder):
    """CADEC is an annoted corpus of patient reported Adverse Drug Events."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="cadec_source",
            version=SOURCE_VERSION,
            description="CADEC source schema",
            schema="source",
            subset_id="cadec",
        ),
        BigBioConfig(
            name="cadec_bigbio_kb",
            version=BIGBIO_VERSION,
            description="CADEC BigBio schema",
            schema="bigbio_kb",
            subset_id="cadec",
        ),
    ]

    DEFAULT_CONFIG_NAME = "cadec_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "text_bound_annotations": [
                        {
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                            "text": datasets.Sequence(datasets.Value("string")),
                            "type": datasets.Value("string"),
                            "id": datasets.Value("string"),
                        }
                    ],
                    "normalizations": [
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "ref_id": datasets.Value("string"),
                            "resource_name": datasets.Value("string"),
                            "cuid": datasets.Value("string"),
                            "text": datasets.Value("string"),
                        }
                    ],
                },
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

        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "corpus_path": os.path.join(data_dir, "cadec"),
                },
            ),
        ]

    def _generate_examples(
        self,
        corpus_path: str,
    ) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        if self.config.schema == "source":
            for guid, example in self._parse_examples(corpus_path):
                example["id"] = str(guid)
                yield guid, example
        elif self.config.schema == "bigbio_kb":
            for guid, example in self._parse_examples(corpus_path):
                example["events"] = []
                example["relations"] = []
                example["equivalences"] = []
                example = parsing.brat_parse_to_bigbio_kb(example)
                example["id"] = str(guid)
                yield guid, example
        else:
            raise ValueError(f"Invalid config: {self.config.name}")

    def _parse_examples(self, corpus_path: Path) -> Dict:

        txt_files = list(Path(corpus_path, "text").glob("*txt"))
        for guid, txt_file in enumerate(txt_files):
            example = {}
            example["document_id"] = txt_file.with_suffix("").name
            with txt_file.open() as f:
                example["text"] = f.read()

            example["text_bound_annotations"] = []
            example["normalizations"] = []
            for annotation in ["original", "meddra", "sct"]:
                annotation_path = Path(
                    corpus_path, annotation, txt_file.with_suffix(".ann").name
                )
                self._populate_example(example, annotation, annotation_path)
            yield guid, example

    def _populate_example(self, example, code_system: str, annotation_file: Path):
        ann_lines = []
        if annotation_file.exists():
            with annotation_file.open(encoding="iso-8859-1") as f:
                ann_lines.extend(f.readlines())

        for line in ann_lines:
            line = line.strip()
            # some lines contain 4 or 5 or even 7 spaces instead of tabs (f.e. /meddra/DICLOFENAC-SODIUM.7.ann)
            line = re.sub(r"\s{4,}", "\t", line)

            if not line:
                continue
            elif line.startswith("TT"):  # Normalization
                fields = line.split("\t")
                ann = self._parse_normalization(fields, code_system)
                example["normalizations"].extend(ann)
            elif line.startswith("T"):  # Text bound
                ann = {}
                fields = line.split("\t")
                ann["id"] = fields[0]
                ann["type"], span_str = fields[1].split(maxsplit=1)
                ann["offsets"] = []
                for span in span_str.split(";"):
                    start, end = span.split()
                    ann["offsets"].append([int(start), int(end)])

                text = fields[2]
                # Heuristically split text of discontiguous entities into chunks
                ann["text"] = []
                if len(ann["offsets"]) > 1:
                    i = 0
                    for start, end in ann["offsets"]:
                        chunk_len = end - start
                        ann["text"].append(text[i : chunk_len + i])
                        i += chunk_len
                        while i < len(text) and text[i] == " ":
                            i += 1
                else:
                    ann["text"] = [text]
                example["text_bound_annotations"].append(ann)
            elif line.startswith("#"):
                continue
            else:
                raise ValueError(f"Invalid tag: {line}")
        return example

    def _parse_normalization(self, fields: List[str], code_system: str):
        anns = []
        base_ann = {
            "ref_id": fields[0][1:],  # here TT1 -> Normalization of T1
            "type": "Reference",
        }

        if "CONCEPT_LESS" in fields[1]:
            ann = base_ann.copy()
            ann["id"] = code_system + "_" + fields[0]
            ann["cuid"], _ = fields[1].split(maxsplit=1)
            ann["text"] = ""
            ann["resource_name"] = ""
            anns = [ann]
        elif code_system == "sct":
            concepts = re.sub(r"\s+", " ", fields[1])
            # strip random amount of whitespace from seperator |
            concepts = re.sub(r"(\s?\|\s?)", "|", concepts)
            # remove all offsets, sometimes there is a | between normalization and offset, sometimes there is not
            # f.e. TT9	21499005|Feeling agitated 232 249	Severe aggitation (LIPITOR.496.ann)
            concepts = re.sub(r"(\s?\|?\s?([0-9]+)\s([0-9]+);?)+", "", concepts)
            # TT7	2297011000036108 | magnesium | (substance) 186 195	magnesium (LIPITOR.598.ANN)
            concepts = re.sub(r"(\s?\|\s?\(substance\))", " (substance)", concepts)
            # separator for multiple normalizations can be '|+’ or ’|or’ with a random amount of whitespace
            # see /sct/ARTHROTEC.112.ann, /sct/ARTHROTEC.113.ann, /sct/ARTHROTEC.1.ann
            concepts = re.sub(r"(\s?\|\s?or\s?)", "|+", concepts)
            concepts = re.sub(r"(\s?\|\s?\+\s?)", "|+", concepts)
            concepts = concepts.split("|+")

            anns = []
            for concept in concepts:
                concept = concept.strip()
                ann = base_ann.copy()
                if "|" in concept:
                    ann["cuid"], ann["text"] = concept.split("|")
                else:
                    # sometimes there is no | between cuid and label
                    # TT2     76948002 | Severe pain | + 57676002 Arthralgia |  43 72 extreme pain \
                    # in all my joints (LIPITOR.698.ann)
                    ann["cuid"], ann["text"] = concept.split(" ")
                ann["id"] = f"{code_system}_{fields[0]}_{ann['cuid']}"
                ann["cuid"] = ann["cuid"].strip()
                assert ann["cuid"].isnumeric(), f"Error: concept {ann['cuid']} is not numeric"
                ann["text"] = ann["text"].strip()
                ann["resource_name"] = "Snomed CT"
                anns.append(ann)
        elif code_system == "meddra":
            # remove offsets
            concepts = re.sub(r"\s(([0-9]+)\s([0-9]+);)*([0-9]+)\s([0-9]+)$", "", fields[1])
            concepts = concepts.split(" + ") # multiple codes are usually split with +
            for concept_ in concepts:
                for concept in concept_.split("/"):  # sometimes with /
                    ann = base_ann.copy()
                    ann["cuid"] = concept
                    assert ann["cuid"].isnumeric(), f"{fields} Error: concept {ann['cuid']} is not numeric"
                    ann["text"] = ""
                    ann["id"] = f"{code_system}_{fields[0]}_{ann['cuid']}"
                    ann["resource_name"] = "Meddra"
                    anns.append(ann)
        return anns
