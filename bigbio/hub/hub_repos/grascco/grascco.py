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
GraSCCo is a collection of artificially generated semi-structured and unstructured German-language clinical summaries.
These summaries are formulated as letters from the hospital to the patient's GP after in-patient or out-patient care.
This is common practice in Germany, Austria and Switzerland.

The creation of the GraSCCo documents were inspired by existing clinical texts,
but all names and dates are purely fictional.
There is no relation to existing patients, clinicians or institutions.
Whereas the texts try to represent the range of German clinical language as best as possible,
medical plausibility must not be assumed.

GraSCCo can therefore only be used to train clinical language models, not clinical domain models.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from .bigbiohub import BigBioConfig, Tasks, kb_features, logger

_LOCAL = False

_CITATION = """\
@incollection{modersohn2022grascco,
  title={GRASCCO—The First Publicly Shareable, Multiply-Alienated German Clinical Text Corpus},
  author={Modersohn, Luise and Schulz, Stefan and Lohr, Christina and Hahn, Udo},
  booktitle={German Medical Data Sciences 2022--Future Medicine: More Precise, More Integrative, More Sustainable!},
  pages={66--72},
  year={2022},
  publisher={IOS Press}
}
"""

_DATASETNAME = "grascco"

_DISPLAYNAME = "GraSCCo"

_DESCRIPTION = """\
GraSCCo is a collection of artificially generated semi-structured and unstructured German-language clinical summaries.
These summaries are formulated as letters from the hospital to the patient's GP after in-patient or out-patient care.
This is common practice in Germany, Austria and Switzerland.

The creation of the GraSCCo documents were inspired by existing clinical texts,
but all names and dates are purely fictional.
There is no relation to existing patients, clinicians or institutions.
Whereas the texts try to represent the range of German clinical language as best as possible,
medical plausibility must not be assumed.

GraSCCo can therefore only be used to train clinical language models, not clinical domain models.
"""

_HOMEPAGE = "https://zenodo.org/records/6539131"

_LICENSE = "CC_BY_4p0"

_LANGUAGES = ["German"]

_PUBMED = False

_URLS = {
    _DATASETNAME: {
        "phi": "https://zenodo.org/records/11502329/files/grascco_phi_annotation_json.zip?download=1",
    },
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"

_UIMA_FEATURES_KEY = "%FEATURE_STRUCTURES"


class GraSCCoDataset(datasets.GeneratorBasedBuilder):
    """Dataloader for GraSCCo dataset with different annotation layers (PHI, SNOMED CT, etc.)"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="grascco_phi_source",
            version=SOURCE_VERSION,
            description="GraSCCo (PHI) source schema",
            schema="source",
            subset_id="phi",
        ),
        BigBioConfig(
            name="grascco_phi_bigbio_kb",
            version=BIGBIO_VERSION,
            description="GraSCCo (PHI) BigBio schema",
            schema="bigbio_kb",
            subset_id="phi",
        ),
    ]

    DEFAULT_CONFIG_NAME = "grascco_phi_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "document_id": datasets.Value("string"),
                    _UIMA_FEATURES_KEY: [
                        {
                            "%ID": datasets.Value("int64"),
                            "%TYPE": datasets.Value("string"),
                            "@sofa": datasets.Value("int64"),
                            "@layer": datasets.Value("int64"),
                            "begin": datasets.Value("int64"),
                            "end": datasets.Value("int64"),
                            "name": datasets.Value("string"),
                            "uiName": datasets.Value("string"),
                            "documentTitle": datasets.Value("string"),
                            "sofaString": datasets.Value("string"),
                        }
                    ],
                }
            )

        elif self.config.schema == "bigbio_kb":
            features = kb_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        urls = _URLS[_DATASETNAME][self.config.subset_id]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": Path(data_dir) / "grascco_phi_annotation_json",
                },
            ),
        ]

    def _parse_uima_cas_json(self, filename) -> Dict:
        """Parse UIMA CAS JSON file and return parsed elements as well as the raw data"""
        with open(filename, "r", encoding="utf-8") as f:
            uima_features = json.load(f)[_UIMA_FEATURES_KEY]
            phi_elements = []
            for feature in uima_features:
                if feature["%TYPE"] == "webanno.custom.PHI":
                    phi_elements.append(feature)
                if feature["%TYPE"] == "de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData":
                    document_title = feature["documentTitle"]
                if feature["%TYPE"] == "uima.cas.Sofa":
                    document_text = feature["sofaString"]
            return {
                "phi_elements": phi_elements,
                "document_title": document_title,
                "document_text": document_text,
                "uima_features": uima_features,
            }

    def _generate_examples(self, filepath) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        for file_id, file in enumerate(sorted(filepath.glob("*.json"))):
            uima_parsed = self._parse_uima_cas_json(file)
            doc_id = uima_parsed["document_title"]
            if self.config.schema == "source":
                yield doc_id, {"document_id": doc_id, _UIMA_FEATURES_KEY: uima_parsed["uima_features"]}
            elif self.config.schema == "bigbio_kb":
                text = uima_parsed["document_text"]
                relations = []
                entities = []
                # Just as single passage; ignoring sentence boundaries from annotation tool, as these are not reliable
                passages = [{"id": f"{file_id}-0", "type": "document", "text": [text], "offsets": [[0, len(text)]]}]

                # Other subsets / annotation layers will be added in future GraSCCo releases
                if self.config.subset_id == "phi":
                    for phi in sorted(uima_parsed["phi_elements"], key=lambda p: p["begin"]):
                        e_start = phi["begin"]
                        e_end = phi["end"]
                        eid = phi["%ID"]
                        if "kind" not in phi:
                            logger.warning(
                                f"'kind' attribute missing in PHI element with ID {eid} in document {doc_id}"
                            )
                            continue
                        entities.append(
                            {
                                "id": f"{file_id}-{eid}",
                                "type": phi["kind"],
                                "text": [text[e_start:e_end]],
                                "offsets": [[e_start, e_end]],
                                "normalized": [],
                            }
                        )

                yield doc_id, {
                    "id": file_id,
                    "document_id": doc_id,
                    "passages": passages,
                    "entities": entities,
                    "events": [],
                    "coreferences": [],
                    "relations": relations,
                }
