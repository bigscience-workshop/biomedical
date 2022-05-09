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

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import datasets

from bigbio.utils import parsing, schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Tasks

_LOCAL = False
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
    "diann_iber_eval_en": {
        "en": "https://github.com/gildofabregat/DIANN-IBEREVAL-2018/raw/master/DIANN_CORPUS/english.zip"
    },
    "diann_iber_eval_es": {
        "es": "https://github.com/gildofabregat/DIANN-IBEREVAL-2018/raw/master/DIANN_CORPUS/spanish.zip"
    },
    "diann_iber_eval": {
        "es": "https://github.com/gildofabregat/DIANN-IBEREVAL-2018/raw/master/DIANN_CORPUS/spanish.zip",
        "en": "https://github.com/gildofabregat/DIANN-IBEREVAL-2018/raw/master/DIANN_CORPUS/english.zip",
    },
}

_SUPPORTED_TASKS = [Tasks.TRANSLATION, Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"

_LANG2CODE = {"English": "en", "Spanish": "es"}
_CODE2LANG = {"en": "English", "es": "Spanish"}


class DIANNIberEvalDataset(datasets.GeneratorBasedBuilder):
    """DIANN's corpus is a collection of 500 abstracts related to the biomedical domain, in Spanish and English"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = []
    BUILDER_CONFIGS.append(
        BigBioConfig(
            name="diann_iber_eval_source",
            version=SOURCE_VERSION,
            description="DIANN Iber Eval source schema",
            schema="source",
            subset_id="diann_iber_eval",
        ),
    )

    BUILDER_CONFIGS.append(
        BigBioConfig(
            name="diann_iber_eval_bigbio_kb",
            version=SOURCE_VERSION,
            description="DIANN Iber Eval BigBio schema",
            schema="bigbio_kb",
            subset_id="diann_iber_eval",
        ),
    )

    for language in ["en", "es"]:
        BUILDER_CONFIGS.append(
            BigBioConfig(
                name=f"diann_iber_eval_{language}_source",
                version=SOURCE_VERSION,
                description=f"DIANN Iber Eval {_CODE2LANG[language]} source schema",
                schema="source",
                subset_id=f"diann_iber_eval_{language}",
            ),
        )

        BUILDER_CONFIGS.append(
            BigBioConfig(
                name=f"diann_iber_eval_{language}_bigbio_kb",
                version=BIGBIO_VERSION,
                description=f"DIANN Iber Eval {_CODE2LANG[language]} BigBio schema",
                schema="bigbio_kb",
                subset_id=f"diann_iber_eval_{language}",
            ),
        )

    BUILDER_CONFIGS.append(
        BigBioConfig(
            name="diann_iber_eval_bigbio_t2t",
            version=BIGBIO_VERSION,
            description="DIANN Iber Eval BigBio translation schema",
            schema="bigbio_t2t",
            subset_id="diann_iber_eval",
        ),
    )

    DEFAULT_CONFIG_NAME = "diann_iber_eval_en_source"

    _ENTITY_TYPES = {
        "Disability",
        "Neg",
        "Scope",
    }

    _STUDIES_PATH = {
        "diann_iber_eval_en": ["English"],
        "diann_iber_eval_es": ["Spanish"],
        "diann_iber_eval": ["English", "Spanish"],
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
                    "language": datasets.Value("string"),
                    "XML_annotation": datasets.Value("string"),
                    "metadata": {
                        "title": datasets.Value("string"),
                        "keywords": datasets.Sequence([datasets.Value("string")]),
                    },
                }
            )

        elif self.config.schema == "bigbio_kb":
            features = schemas.kb_features

        elif self.config.schema == "bigbio_t2t":
            features = schemas.text2text_features

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
        root_dirs = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"root_dirs": root_dirs, "split": "Training"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"root_dirs": root_dirs, "split": "Test"},
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

    def _generate_source_examples(self, root_dirs: Dict, split: str) -> Tuple[int, Dict]:
        _id = -1
        for lang_code, root_dir in root_dirs.items():
            print(lang_code, root_dir)
            language = _CODE2LANG[lang_code]

            dir_files = Path(root_dir) / language / f"{language}_{split}"
            raw_dir = Path(root_dir) / language / f"{language}_{split}" / "Raw"
            raw_files = list(raw_dir.glob("*txt"))

            for raw_file in raw_files:
                raw_text = raw_file.read_text()
                uid = raw_file.stem

                xml_annotation, entities = self._get_annotations_from_uid(dir_files, uid)
                metadata = self._get_metadata_from_uid(dir_files, uid)
                _id += 1

                yield _id, {
                    "id": _id,
                    "doc_id": uid,
                    "text": raw_text,
                    "entities": entities,
                    "language": _LANG2CODE[language],
                    "XML_annotation": xml_annotation,
                    "metadata": metadata,
                }

    def _generate_bigbio_kb_examples(self, root_dirs: Dict, split: str) -> Tuple[int, Dict]:
        _id = -1
        for lang_code, root_dir in root_dirs.items():
            language = _CODE2LANG[lang_code]

            dir_files = Path(root_dir) / language / f"{language}_{split}"
            raw_dir = Path(root_dir) / language / f"{language}_{split}" / "Raw"
            raw_files = list(raw_dir.glob("*txt"))

            for raw_file in raw_files:
                raw_text = raw_file.read_text()
                uid = raw_file.stem

                _, entities = self._get_annotations_from_uid(dir_files, uid)
                _id += 1

                example = parsing.brat_parse_to_bigbio_kb(
                    {
                        "document_id": f"{uid}_{lang_code}",
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

                example["id"] = _id
                yield _id, example

    def _generate_bigbio_t2t_examples(self, root_dirs: Dict, split: str) -> Tuple[int, Dict]:
        sample_map = defaultdict(list)
        for lang_code, root_dir in root_dirs.items():
            language = _CODE2LANG[lang_code]
            raw_dir = Path(root_dir) / language / f"{language}_{split}" / "Raw"
            raw_files = list(raw_dir.glob("*txt"))

            for raw_file in raw_files:
                raw_text = raw_file.read_text()
                uid = raw_file.stem
                sample_map[uid].append({"language": lang_code, "text": raw_text})

        _id = -1
        for sample_id_prefix, sample_pair in sample_map.items():
            if len(sample_pair) != 2:
                continue
            en_idx = 0 if sample_pair[0]["language"] == "en" else 1
            es_idx = 0 if en_idx == 1 else 1

            _id += 1
            yield _id, {
                "id": sample_id_prefix,
                "document_id": sample_id_prefix,
                "text_1": sample_pair[en_idx]["text"],
                "text_2": sample_pair[es_idx]["text"],
                "text_1_name": "en",
                "text_2_name": "es",
            }

    def _generate_examples(self, root_dirs: Dict, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        if self.config.schema == "source":
            examples = self._generate_source_examples(root_dirs, split)
        elif self.config.schema == "bigbio_kb":
            examples = self._generate_bigbio_kb_examples(root_dirs, split)
        elif self.config.schema == "bigbio_t2t":
            examples = self._generate_bigbio_t2t_examples(root_dirs, split)

        for _id, sample in examples:
            yield _id, sample
