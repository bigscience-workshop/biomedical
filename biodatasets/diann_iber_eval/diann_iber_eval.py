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
DIANN's corpus consists of a collection of 500 abstracts from Elsevier journal papers
related to the biomedical domain, where both Spanish and English versions are available,
with annotations for disabilities appearing in these abstracts.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from utils import parsing, schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

_CITATION = """\
@proceedings{DBLP:conf/sepln/2018ibereval,
  author    = {Paolo Rosso and
               Julio Gonzalo and
               Raquel Martinez and
               Soto Montalvo and
               Jorge Carrillo de Albornoz},
  title     = {Proceedings of the Third Workshop on Evaluation of Human Language
               Technologies for Iberian Languages (IberEval 2018) co-located with
               34th Conference of the Spanish Society for Natural Language Processing
               {(SEPLN} 2018), Sevilla, Spain, September 18th, 2018},
  series    = {{CEUR} Workshop Proceedings},
  publisher = {CEUR-WS.org},
  volume    = {2150},
  year      = {2018},
  url       = {http://ceur-ws.org/Vol-2150},
  biburl    = {https://dblp.org/rec/conf/sepln/2018ibereval.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_DATASETNAME = "diann_iber_eval"

_DESCRIPTION = """\
DIANN's corpus consists of a collection of 500 abstracts from Elsevier journal papers
related to the biomedical domain, where both Spanish and English versions are available.

This dataset contains annotations for disabilities appearing in these abstracts,
usually expressed either with a specific word, such as "blindness", or as the limitation
or lack of a human function, such as "lack of vision".

(Spanish)
El corpus DIANN se compone de una colección de 500 resúmenes de artículos de revista Elsevier
del ámbito biomédico, con versiones en español e inglés.

Este conjunto de datos contiene anotaciones para discapacidades que aparecen en dichos resúmenes,
expresadas por medio de palabras específicas, como "ablepsia", or como la limitación o falta de
una función, como "falta de visión".
"""

_HOMEPAGE = "http://nlp.uned.es/diann/"

_LICENSE = "UNKNOWN"

_URLS = {
    "diannibereval_en": "https://github.com/gildofabregat/DIANN-IBEREVAL-2018/raw/master/DIANN_CORPUS/english.zip",
    "diannibereval_sp": "https://github.com/gildofabregat/DIANN-IBEREVAL-2018/raw/master/DIANN_CORPUS/spanish.zip",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]  # TODO: also for translation?

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class DIANNIberEvalDataset(datasets.GeneratorBasedBuilder):
    """DIANN's corpus is a collection of 500 abstracts related to the biomedical domain, in Spanish and English"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = []
    for language in ["en", "sp"]:
        BUILDER_CONFIGS.append(
            BigBioConfig(
                name=f"diannibereval_{language}_source",
                version=SOURCE_VERSION,
                description=f"DIANN Iber Eval {language.capitalize()} source schema",
                schema="source",
                subset_id=f"diannibereval_{language}",
            )
        )
        BUILDER_CONFIGS.append(
            BigBioConfig(
                name=f"diannibereval_{language}_bigbio_kb",
                version=BIGBIO_VERSION,
                description=f"DIANN Iber Eval {language.capitalize()} BigBio schema",
                schema="bigbio_kb",
                subset_id=f"diannibereval_{language}",
            ),
        )

    DEFAULT_CONFIG_NAME = "diannibereval_en_source"

    _ENTITY_TYPES = {
        "Disability",
        "Neg",
        "Scope",
    }

    def _info(self) -> datasets.DatasetInfo:
        """
        Provide information about DIANN Iber Eval 2018
        """

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "doc_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "entities": [
                        {
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                            "text": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "id": datasets.Value("string"),
                        }
                    ],
                    "XML_annotation": datasets.Value("string"),
                    "metadata": {
                        "title": datasets.Value("string"),
                        "keywords": datasets.Sequence([datasets.Value("string")]),
                    },
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
        urls = _URLS[self.config.subset_id]
        data_dir = Path(dl_manager.download_and_extract(urls))
        studies_path = {
            "diannibereval_en": "English",
            "diannibereval_sp": "Spanish",
        }

        study_path = studies_path[self.config.subset_id]

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"dir_files": data_dir / study_path / f"{study_path}_Training"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"dir_files": data_dir / study_path / f"{study_path}_Test"},
            ),
        ]

    def _get_annotations_from_uid(self, dir_files: Path, uid: str) -> Tuple[str, Dict]:
        """Adapted from https://github.com/diannibereval2018/evaluation/blob/master/utils/brat.py
        (official repo of DIANN Iber Eval 2018)

        Args:
            dir_files (Path): Training or Test path containing directories Annotated, Metadata and Raw
            uid (str): unique identifier of a text sample

        Returns:
            Tuple[str, Dict]: this tuple contains (0) the annotation with raw XML tags
                              and (1) a dictionary with a reconstruction of the BigBio
                              schema
        """
        example = {}

        example["text_bound_annotations"] = []
        example["events"] = []  # unused
        example["relations"] = []  # unused
        example["equivalences"] = []  # unused
        example["attributes"] = []  # unused
        example["normalizations"] = []  # unused

        xml_file = dir_files / "Annotated" / f"{uid}.txt"
        xml_str = xml_file.read_text()

        T_uid = 1

        pre_to_translate = {
            "dis": ["Disability", "T", (0, 0)],
            "neg": ["Neg", "T", (0, 0)],
            "scp": ["Scope", "T", (0, 0)],
        }

        i, x, z, state = 1, 0, 0, 0
        inclu = []
        chest = dict()

        for charac in xml_str:
            if state == 0:
                _s_tag_open, _e_tag_open = x + 1, x + 4
                _s_tag_close, _e_tag_close = x, x + 2
                _s_tag_type, _e_tag_type = x + 2, x + 5
                if charac == "<" and len(xml_str) > x + 4 and xml_str[_s_tag_open:_e_tag_open] in pre_to_translate:
                    if not xml_str[_s_tag_open:_e_tag_open] in chest:
                        chest[xml_str[_s_tag_open:_e_tag_open]] = list()
                    chest[xml_str[_s_tag_open:_e_tag_open]].append({"text": "", "start": x - z, "end": -1})
                    state = 4
                    inclu.append(True)
                    z += 1
                elif (
                    len(xml_str) > x + 5
                    and xml_str[_s_tag_close:_e_tag_close] == "</"
                    and xml_str[_s_tag_type:_e_tag_type] in pre_to_translate
                ):
                    for d in chest[xml_str[_s_tag_type:_e_tag_type]]:
                        if d["end"] < 0:
                            d["end"] = x - z
                    state = 5
                    inclu.remove(True)
                    z += 1
                elif any(inclu):
                    for c in chest:
                        for d in chest[c]:
                            if d["end"] < 0:
                                d["text"] += charac
                    i += 1
            else:
                state -= 1
                z += 1
            x += 1

        example = []
        for classes in chest:
            for i, terms in enumerate(chest[classes]):
                example.append(
                    {
                        "offsets": [[terms["start"], terms["end"]]],
                        "text": [terms["text"]],
                        "type": pre_to_translate[classes][0],
                        "id": f"{pre_to_translate[classes][1]}{T_uid}",
                    }
                )

                T_uid += 1

        return xml_str, example

    def _get_metadata_from_uid(self, dir_files: Path, uid: str) -> Dict[str, List[str]]:
        """
        Args:
            dir_files (Path): Training or Test path containing directories Annotated, Metadata and Raw
            uid (str): unique identifier of a text sample

        Returns:
            Dict[str, List[str]]: a dictionary with the "title" (contained in Metadata/{UID}_title.txt),
                                  and the "keywords" (contained in Metadata/{UID}_keywords.txt, one per line)
        """

        keyword_file = dir_files / "Metadata" / f"{uid}_keywords.txt"
        title_file = dir_files / "Metadata" / f"{uid}_title.txt"

        keyword_str = keyword_file.read_text()
        title_str = title_file.read_text()

        keyword_indicators = ["key words", "keywords"]

        keyword_list = [
            line for line in keyword_str.split("\n") if line.strip() != "" and line.lower() not in keyword_indicators
        ]

        return {"title": title_str, "keywords": keyword_list}

    def _generate_examples(self, dir_files) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        raw_dir = dir_files / "Raw"
        raw_files = list(raw_dir.glob("*txt"))

        for idx, raw_file in enumerate(raw_files):
            raw_text = raw_file.read_text()
            uid = raw_file.stem

            xml_annotation, entities = self._get_annotations_from_uid(dir_files, uid)
            metadata = self._get_metadata_from_uid(dir_files, uid)

            if self.config.schema == "source":
                yield idx, {
                    "id": idx,
                    "doc_id": uid,
                    "text": raw_text,
                    "entities": entities,
                    "XML_annotation": xml_annotation,
                    "metadata": metadata,
                }
            elif self.config.schema == "bigbio_kb":
                example = parsing.brat_parse_to_bigbio_kb(
                    {
                        "document_id": uid,
                        "text": raw_text,
                        "text_bound_annotations": entities,
                        "normalizations": [],
                        "events": [],
                        "relations": [],
                        "equivalences": [],
                        "attributes": [],
                    },
                    entity_types=self._ENTITY_TYPES,
                )
                example["id"] = idx
                yield idx, example
