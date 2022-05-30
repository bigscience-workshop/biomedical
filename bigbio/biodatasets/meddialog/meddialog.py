# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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

import json
import re

import datasets

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks

_DATASETNAME = "meddialog"

_LANGUAGES = [Lang.EN, Lang.ZH]
_PUBMED = False
_LOCAL = False
_CITATION = """
@article{DBLP:journals/corr/abs-2004-03329,
  author    = {Shu Chen and
               Zeqian Ju and
               Xiangyu Dong and
               Hongchao Fang and
               Sicheng Wang and
               Yue Yang and
               Jiaqi Zeng and
               Ruisi Zhang and
               Ruoyu Zhang and
               Meng Zhou and
               Penghui Zhu and
               Pengtao Xie},
  title     = {MedDialog: {A} Large-scale Medical Dialogue Dataset},
  journal   = {CoRR},
  volume    = {abs/2004.03329},
  year      = {2020},
  url       = {https://arxiv.org/abs/2004.03329},
  eprinttype = {arXiv},
  eprint    = {2004.03329},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2004-03329.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_DESCRIPTION = """
The MedDialog dataset (English) contains conversations (in English) between doctors and patients.\
It has 0.26 million dialogues. The data is continuously growing and more dialogues will be added. \
The raw dialogues are from healthcaremagic.com and icliniq.com.\

All copyrights of the data belong to healthcaremagic.com and icliniq.com.
"""

_HOMEPAGE = "https://github.com/UCSD-AI4H/Medical-Dialogue-System"

_LICENSE = "Public for Research"

_URLs = {
    "en": {
        "train": "https://drive.google.com/file/d/1ria4E6IdTIPsikL4Glm3uy1tFKJKw0W8/view?usp=sharing",
        "validation": "https://drive.google.com/file/d/1KAZneuwdfEVQQM6euCX4pMDP-9DQpiB5/view?usp=sharing",
        "test": "https://drive.google.com/file/d/10izqL71kcgnteYsf87Vh6j_mZ8sZM2Rc/view?usp=sharing",
    },
    "zh": {
        "train": "https://drive.google.com/file/d/1AaDJoHaiHAwEZwtskRH8oL1UP4FRgmgx/view?usp=sharing",
        "validation": "https://drive.google.com/file/d/1TvfZCmQqP1kURIfEinOcj5VOPelTuGwI/view?usp=sharing",
        "test": "https://drive.google.com/file/d/1pmmG95Yl6mMXRXDDSRb9-bYTxOE7ank5/view?usp=sharing",
    },
}

_SUPPORTED_TASKS = [Tasks.TEXT_CLASSIFICATION]
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class MedDialog(datasets.GeneratorBasedBuilder):
    """MedDialog: Large-scale Medical Dialogue Datasets in English and Chinese."""

    DEFAULT_CONFIG_NAME = "meddialog_en_source"
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        # Source schemas
        BigBioConfig(
            name="meddialog_en_source",
            version=SOURCE_VERSION,
            description="MedDialog source schema",
            schema="source",
            subset_id="meddialog_en",
        ),
        BigBioConfig(
            name="meddialog_zh_source",
            version=SOURCE_VERSION,
            description="MedDialog source schema",
            schema="source",
            subset_id="meddialog_zh",
        ),
        # BigBio schema: text classification
        BigBioConfig(
            name="meddialog_en_bigbio_text",
            version=BIGBIO_VERSION,
            description="MedDialog simplified BigBio schema",
            schema="bigbio_text",
            subset_id="meddialog_en",
        ),
        BigBioConfig(
            name="meddialog_zh_bigbio_text",
            version=BIGBIO_VERSION,
            description="MedDialog simplified BigBio schema",
            schema="bigbio_text",
            subset_id="meddialog_zh",
        ),
    ]

    def _get_gdrive_url(self, url):
        """Converts URL from google drive shareable link to format used by dl_manager."""
        fileid = re.match("https://drive\.google\.com/file/d/(.+)/view\?", url).group(1)
        return f"https://drive.google.com/uc?id={fileid}"

    def _info(self):
        lang = self.config.name.split("_")[1]
        if self.config.schema == "source":
            if lang == "en":
                features = datasets.Features(
                    {
                        "description": datasets.Value("string"),
                        "utterances": datasets.Sequence(
                            {
                                "speaker": datasets.ClassLabel(
                                    names=["patient", "doctor"]
                                ),
                                "utterance": datasets.Value("string"),
                            }
                        ),
                    }
                )
            elif lang == "zh":
                features = datasets.Features(
                    {
                        "utterances": datasets.Sequence(
                            {
                                "speaker": datasets.ClassLabel(names=["病人", "医生"]),
                                "utterance": datasets.Value("string"),
                            }
                        ),
                    }
                )
        elif self.config.schema == "bigbio_text":
            features = schemas.text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        lang = self.config.name.split("_")[1]
        my_urls = {
            split: self._get_gdrive_url(url) for split, url in _URLs[lang].items()
        }
        dl_dir = dl_manager.download_and_extract(my_urls)
        return [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs={"filepath": dl_dir[split], "split": split, "lang": lang},
            )
            for split in _URLs[lang]
        ]

    def _generate_examples(self, filepath, split, lang):
        with open(filepath, "r") as f:
            data = json.load(f)

        # delimiter symbol differs by language
        delimiter = "：" if lang == "zh" else ":"
        document_id = f"{lang}_{split}"

        for i, d in enumerate(data):
            out_utterances = []
            utterances = d["utterances"] if lang == "en" else d
            for j, utt in enumerate(utterances):
                elements = utt.strip().split(delimiter)
                speaker = elements[0]
                text = delimiter.join(elements[1:]).strip()
                if self.config.schema == "bigbio_text":
                    # TODO - this ignores description
                    id = f"{document_id}_{i}_{j}"
                    yield id, {
                        "id": id,
                        "document_id": document_id,
                        "text": text,
                        "labels": [speaker],
                    }
                else:
                    out_utterances.append({"speaker": speaker, "utterance": text})
            if self.config.schema == "source":
                id = f"{document_id}_{i}"
                if lang == "en":
                    yield id, {
                        "description": d["description"],
                        "utterances": out_utterances,
                    }
                else:
                    yield id, {
                        "utterances": out_utterances,
                    }
