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

import ast
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from bigbio.utils import parsing, schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Tasks
from bigbio.utils.license import Licenses

_LOCAL = False
_CITATION = """\
@article{10.1093/jamia/ocv037,
    author = {Kors, Jan A and Clematide, Simon and Akhondi,
    Saber A and van Mulligen, Erik M and Rebholz-Schuhmann, Dietrich},
    title = "{A multilingual gold-standard corpus for biomedical concept recognition: the Mantra GSC}",
    journal = {Journal of the American Medical Informatics Association},
    volume = {22},
    number = {5},
    pages = {948-956},
    year = {2015},
    month = {05},
    abstract = "{Objective To create a multilingual gold-standard corpus for biomedical concept recognition.Materials
    and methods We selected text units from different parallel corpora (Medline abstract titles, drug labels,
    biomedical patent claims) in English, French, German, Spanish, and Dutch. Three annotators per language
    independently annotated the biomedical concepts, based on a subset of the Unified Medical Language System and
    covering a wide range of semantic groups. To reduce the annotation workload, automatically generated
    preannotations were provided. Individual annotations were automatically harmonized and then adjudicated, and
    cross-language consistency checks were carried out to arrive at the final annotations.Results The number of final
    annotations was 5530. Inter-annotator agreement scores indicate good agreement (median F-score 0.79), and are
    similar to those between individual annotators and the gold standard. The automatically generated harmonized
    annotation set for each language performed equally well as the best annotator for that language.Discussion The use
    of automatic preannotations, harmonized annotations, and parallel corpora helped to keep the manual annotation
    efforts manageable. The inter-annotator agreement scores provide a reference standard for gauging the performance
    of automatic annotation techniques.Conclusion To our knowledge, this is the first gold-standard corpus for
    biomedical concept recognition in languages other than English. Other distinguishing features are the wide variety
    of semantic groups that are being covered, and the diversity of text genres that were annotated.}",
    issn = {1067-5027},
    doi = {10.1093/jamia/ocv037},
    url = {https://doi.org/10.1093/jamia/ocv037},
    eprint = {https://academic.oup.com/jamia/article-pdf/22/5/948/34146393/ocv037.pdf},
}
"""

_DATASETNAME = "mantra_gsc"

_DESCRIPTION = """\
We selected text units from different parallel corpora (Medline abstract titles, drug labels, biomedical patent claims)
in English, French, German, Spanish, and Dutch. Three annotators per language independently annotated the biomedical
concepts, based on a subset of the Unified Medical Language System and covering a wide range of semantic groups.
"""

_HOMEPAGE = "https://biosemantics.erasmusmc.nl/index.php/resources/mantra-gsc"

_LICENSE = Licenses.CC_BY_4p0

_URLS = {
    _DATASETNAME: "http://biosemantics.org/MantraGSC/Mantra-GSC.zip",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.NAMED_ENTITY_DISAMBIGUATION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"

_LANGUAGES = {
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "nl": "Dutch",
    "en": "English",
}

_DATASET_TYPES = {
    "emea": "EMEA",
    "medline": "Medline",
    "patents": "Patents",
}


class MantraGSCDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = []

    for language, dataset_type in product(_LANGUAGES, _DATASET_TYPES):
        if dataset_type == "patents" and language in ["nl", "es"]:
            continue

        BUILDER_CONFIGS.append(
            BigBioConfig(
                name=f"mantra_gsc_{language}_{dataset_type}_source",
                version=SOURCE_VERSION,
                description=f"Mantra GSC {_LANGUAGES[language]} {_DATASET_TYPES[dataset_type]} source schema",
                schema="source",
                subset_id=f"mantra_gsc_{language}_{_DATASET_TYPES[dataset_type]}",
            )
        )
        BUILDER_CONFIGS.append(
            BigBioConfig(
                name=f"mantra_gsc_{language}_{dataset_type}_bigbio_kb",
                version=SOURCE_VERSION,
                description=f"Mantra GSC {_LANGUAGES[language]} {_DATASET_TYPES[dataset_type]} BigBio schema",
                schema="bigbio_kb",
                subset_id=f"mantra_gsc_{language}_{_DATASET_TYPES[dataset_type]}",
            )
        )

    DEFAULT_CONFIG_NAME = "mantra_gsc_en_medline_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "document_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "entities": [
                        {
                            "entity_id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                            "text": datasets.Sequence(datasets.Value("string")),
                            "cui": datasets.Value("string"),
                            "preferred_term": datasets.Value("string"),
                            "semantic_type": datasets.Value("string"),
                            "normalized": [
                                {
                                    "db_name": datasets.Value("string"),
                                    "db_id": datasets.Value("string"),
                                }
                            ],
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
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)
        data_dir = Path(data_dir) / "Mantra-GSC"

        language, dataset_type = self.config.name.split("_")[2:4]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_dir": data_dir,
                    "language": language,
                    "dataset_type": dataset_type,
                },
            ),
        ]

    def _generate_examples(
        self, data_dir: Path, language: str, dataset_type: str
    ) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        data_dir = data_dir / f"{_LANGUAGES[language]}"

        if dataset_type in ["patents", "emea"]:
            data_dir = data_dir / f"{_DATASET_TYPES[dataset_type]}_ec22-cui-best_man"
        else:
            # It is Medline now
            if language != "en":
                data_dir = (
                    data_dir
                    / f"{_DATASET_TYPES[dataset_type]}_EN_{language.upper()}_ec22-cui-best_man"
                )
            else:
                data_dir = [
                    data_dir
                    / f"{_DATASET_TYPES[dataset_type]}_EN_{_lang.upper()}_ec22-cui-best_man"
                    for _lang in _LANGUAGES
                    if _lang != "en"
                ]

        if not isinstance(data_dir, list):
            data_dir: List[Path] = [data_dir]

        raw_files = [raw_file for _dir in data_dir for raw_file in _dir.glob("*.txt")]

        if self.config.schema == "source":
            for i, raw_file in enumerate(raw_files):
                brat_example = parsing.parse_brat_file(raw_file, parse_notes=True)
                source_example = self._to_source_example(brat_example)
                yield i, source_example

        elif self.config.schema == "bigbio_kb":
            for i, raw_file in enumerate(raw_files):
                brat_example = parsing.parse_brat_file(raw_file, parse_notes=True)
                brat_to_bigbio_example = self._brat_to_bigbio_example(brat_example)
                kb_example = parsing.brat_parse_to_bigbio_kb(brat_to_bigbio_example)
                kb_example["id"] = i
                yield i, kb_example

    def _to_source_example(self, brat_example: Dict) -> Dict:
        source_example = {
            "document_id": brat_example["document_id"],
            "text": brat_example["text"],
        }

        source_example["entities"] = []
        for entity_annotation, ann_notes in zip(
            brat_example["text_bound_annotations"], brat_example["notes"]
        ):
            entity_ann = entity_annotation.copy()

            # Change id property name
            entity_ann["entity_id"] = entity_ann["id"]
            entity_ann.pop("id")

            # Get values from annotator notes
            assert entity_ann["entity_id"] == ann_notes["ref_id"]
            notes_values = ast.literal_eval(ann_notes["text"])
            if len(notes_values) == 4:
                cui, preferred_term, semantic_type, semantic_group = notes_values
            else:
                preferred_term, semantic_type, semantic_group = notes_values
                cui = entity_ann["type"]
            entity_ann["cui"] = cui
            entity_ann["preferred_term"] = preferred_term
            entity_ann["semantic_type"] = semantic_type
            entity_ann["type"] = semantic_group
            entity_ann["normalized"] = [{"db_name": "UMLS", "db_id": cui}]

            # Add entity annotation to sample
            source_example["entities"].append(entity_ann)

        return source_example

    def _brat_to_bigbio_example(self, brat_example: Dict) -> Dict:
        kb_example = {
            "document_id": brat_example["document_id"],
            # "unit_id": unit_id,
            "text": brat_example["text"],
        }
        kb_example["text_bound_annotations"] = []
        kb_example["normalizations"] = []
        for entity_annotation, ann_notes in zip(
            brat_example["text_bound_annotations"], brat_example["notes"]
        ):
            entity_ann = entity_annotation.copy()
            # Get values from annotator notes
            assert entity_ann["id"] == ann_notes["ref_id"]
            notes_values = ast.literal_eval(ann_notes["text"])
            if len(notes_values) == 4:
                cui, _, _, semantic_group = notes_values
            else:
                _, _, semantic_group = notes_values
                cui = entity_ann["type"]
            entity_ann["type"] = semantic_group
            kb_example["text_bound_annotations"].append(entity_ann)
            kb_example["normalizations"].append(
                {
                    "type": semantic_group,
                    "ref_id": entity_ann["id"],
                    "resource_name": "UMLS",
                    "cuid": cui,
                    "text": "",
                }
            )

        kb_example["events"] = brat_example["events"]
        kb_example["relations"] = brat_example["relations"]
        kb_example["equivalences"] = brat_example["equivalences"]
        kb_example["attributes"] = brat_example["attributes"]
        kb_example["notes"] = brat_example["notes"]

        return kb_example
