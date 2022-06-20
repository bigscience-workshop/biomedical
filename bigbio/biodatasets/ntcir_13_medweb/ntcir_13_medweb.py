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
NTCIR-13 MedWeb (Medical Natural Language Processing for Web Document) task requires
to perform a multi-label classification that labels for eight diseases/symptoms must
be assigned to each tweet. Given pseudo-tweets, the output are Positive:p or Negative:n
labels for eight diseases/symptoms. The achievements of this task can almost be
directly applied to a fundamental engine for actual applications.

This task provides pseudo-Twitter messages in a cross-language and multi-label corpus,
covering three languages (Japanese, English, and Chinese), and annotated with eight
labels such as influenza, diarrhea/stomachache, hay fever, cough/sore throat, headache,
fever, runny nose, and cold.

The dataset consists of a single archive file:
    - ntcir13_MedWeb_taskdata.zip

which can be obtained after filling out a form to provide information about the
usage context under this URL: http://www.nii.ac.jp/dsc/idr/en/ntcir/ntcir.html

The zip archive contains a folder with name 'MedWeb_TestCollection'.
Inside this folder, there are the following individual data files:
├── NTCIR-13_MedWeb_en_test.xlsx
├── NTCIR-13_MedWeb_en_training.xlsx
├── NTCIR-13_MedWeb_ja_test.xlsx
├── NTCIR-13_MedWeb_ja_training.xlsx
├── NTCIR-13_MedWeb_zh_test.xlsx
└── NTCIR-13_MedWeb_zh_training.xlsx

The excel sheets contain a training and test split for each of the languages
('en' stands for 'english', 'ja' stands for 'japanese' and 'zh' stands for
(simplified) chinese).

The archive file containing this dataset must be on the users local machine
in a single directory that is passed to `datasets.load_dataset` via
the `data_dir` kwarg. This loader script will read this archive file
directly (i.e. the user should not uncompress, untar or unzip any of
the files).

For more information on this dataset, see:
http://research.nii.ac.jp/ntcir/permission/ntcir-13/perm-en-MedWeb.html
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
import pandas as pd

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tags, Tasks
from bigbio.utils.license import Licenses

_TAGS = [Tags.DISEASE, Tags.SENTIMENT_ANALYSIS]
_LANGUAGES = [Lang.EN, Lang.ZH, Lang.JA]
_PUBMED = False
_LOCAL = True
_CITATION = """\
@article{,
  author    = {Shoko Wakamiya, Mizuki Morita, Yoshinobu Kano, Tomoko Ohkuma and Eiji Aramaki},
  title     = {Overview of the NTCIR-13 MedWeb Task},
  journal   = {Proceedings of the 13th NTCIR Conference on Evaluation of Information Access Technologies (NTCIR-13)},
  year      = {2017},
  url       = {
    http://research.nii.ac.jp/ntcir/workshop/OnlineProceedings13/pdf/ntcir/01-NTCIR13-OV-MEDWEB-WakamiyaS.pdf
  },
}
"""

_DATASETNAME = "ntcir_13_medweb"

_DESCRIPTION = """\
NTCIR-13 MedWeb (Medical Natural Language Processing for Web Document) task requires
to perform a multi-label classification that labels for eight diseases/symptoms must
be assigned to each tweet. Given pseudo-tweets, the output are Positive:p or Negative:n
labels for eight diseases/symptoms. The achievements of this task can almost be
directly applied to a fundamental engine for actual applications.

This task provides pseudo-Twitter messages in a cross-language and multi-label corpus,
covering three languages (Japanese, English, and Chinese), and annotated with eight
labels such as influenza, diarrhea/stomachache, hay fever, cough/sore throat, headache,
fever, runny nose, and cold.

For more information, see:
http://research.nii.ac.jp/ntcir/permission/ntcir-13/perm-en-MedWeb.html

As this dataset also provides a parallel corpus of pseudo-tweets for english,
japanese and chinese it can also be used to train translation models between
these three languages.
"""

_HOMEPAGE = "http://research.nii.ac.jp/ntcir/permission/ntcir-13/perm-en-MedWeb.html"

_LICENSE = Licenses.CC_BY_4p0

# NOTE: Data can only be obtained (locally) by first filling out form to provide
# information about usage context under this link: http://www.nii.ac.jp/dsc/idr/en/ntcir/ntcir.html
_URLS = {
    _DATASETNAME: "ntcir13_MedWeb_taskdata.zip",
}

_SUPPORTED_TASKS = [
    Tasks.TRANSLATION,
    Tasks.TEXT_CLASSIFICATION,
]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class NTCIR13MedWebDataset(datasets.GeneratorBasedBuilder):
    """
    NTCIR-13 MedWeb (Medical Natural Language Processing for Web Document) task requires
    to perform a multi-label classification that labels for eight diseases/symptoms must
    be assigned to each tweet. Given pseudo-tweets, the output are Positive:p or Negative:n
    labels for eight diseases/symptoms. The achievements of this task can almost be
    directly applied to a fundamental engine for actual applications.

    This task provides pseudo-Twitter messages in a cross-language and multi-label corpus,
    covering three languages (Japanese, English, and Chinese), and annotated with eight
    labels such as influenza, diarrhea/stomachache, hay fever, cough/sore throat, headache,
    fever, runny nose, and cold.

    For more information, see:
    http://research.nii.ac.jp/ntcir/permission/ntcir-13/perm-en-MedWeb.html

    As this dataset also provides a parallel corpus of pseudo-tweets for english,
    japanese and chinese it can also be used to train translation models between
    these three languages.
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        # Source configuration - all classification data for all languages
        BigBioConfig(
            name="ntcir_13_medweb_source",
            version=SOURCE_VERSION,
            description="NTCIR 13 MedWeb source schema",
            schema="source",
            subset_id="ntcir_13_medweb_source",
        )
    ]
    for language_name, language_code in (
        ("Japanese", "ja"),
        ("English", "en"),
        ("Chinese", "zh"),
    ):
        # NOTE: BigBio text classification configurations
        # Text classification data for each language
        BUILDER_CONFIGS.append(
            BigBioConfig(
                name=f"ntcir_13_medweb_classification_{language_code}_bigbio_text",
                version=BIGBIO_VERSION,
                description=f"NTCIR 13 MedWeb BigBio {language_name} Classification schema",
                schema="bigbio_text",
                subset_id=f"ntcir_13_medweb_classification_{language_code}_bigbio_text",
            ),
        )

        for target_language_name, target_language_code in (
            ("Japanese", "ja"),
            ("English", "en"),
            ("Chinese", "zh"),
        ):
            # NOTE: BigBio text to text (translation) configurations
            # Parallel text corpora for all pairs of languages
            if language_name != target_language_name:
                BUILDER_CONFIGS.append(
                    BigBioConfig(
                        name=f"ntcir_13_medweb_translation_{language_code}_{target_language_code}_bigbio_t2t",
                        version=BIGBIO_VERSION,
                        description=(
                            f"NTCIR 13 MedWeb BigBio {language_name} -> {target_language_name} translation schema",
                        ),
                        schema="bigbio_t2t",
                        subset_id=f"ntcir_13_medweb_translation_{language_code}_{target_language_code}_bigbio_t2t",
                    ),
                )

    DEFAULT_CONFIG_NAME = "ntcir_13_medweb_source"

    def _info(self) -> datasets.DatasetInfo:
        # Create the source schema; this schema will keep all keys/information/labels
        # as close to the original dataset as possible.
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "ID": datasets.Value("string"),
                    "Language": datasets.Value("string"),
                    "Tweet": datasets.Value("string"),
                    "Influenza": datasets.Value("string"),
                    "Diarrhea": datasets.Value("string"),
                    "Hayfever": datasets.Value("string"),
                    "Cough": datasets.Value("string"),
                    "Headache": datasets.Value("string"),
                    "Fever": datasets.Value("string"),
                    "Runnynose": datasets.Value("string"),
                    "Cold": datasets.Value("string"),
                }
            )
        elif self.config.schema == "bigbio_text":
            features = schemas.text_features
        elif self.config.schema == "bigbio_t2t":
            features = schemas.text2text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        if self.config.data_dir is None:
            raise ValueError(
                "This is a local dataset. Please pass the data_dir kwarg to load_dataset."
            )
        else:
            data_dir = self.config.data_dir

        raw_data_dir = dl_manager.download_and_extract(
            str(Path(data_dir) / _URLS[_DATASETNAME])
        )

        data_dir = Path(raw_data_dir) / "MedWeb_TestCollection"

        if self.config.schema == "source":
            filepaths = {
                datasets.Split.TRAIN: sorted(Path(data_dir).glob("*_training.xlsx")),
                datasets.Split.TEST: sorted(Path(data_dir).glob("*_test.xlsx")),
            }
        elif self.config.schema == "bigbio_text":
            # NOTE: Identify the language for the chosen subset using regex
            pattern = r"ntcir_13_medweb_classification_(?P<language_code>ja|en|zh)_bigbio_text"
            match = re.search(pattern=pattern, string=self.config.subset_id)

            if not match:
                raise ValueError(
                    "Unable to parse language code for text classification from dataset subset id: "
                    f"'{self.config.subset_id}'. Attempted to parse using this regex pattern: "
                    f"'{pattern}' but failed to get a match."
                )

            language_code = match.group("language_code")

            filepaths = {
                datasets.Split.TRAIN: (
                    Path(data_dir) / f"NTCIR-13_MedWeb_{language_code}_training.xlsx",
                ),
                datasets.Split.TEST: (
                    Path(data_dir) / f"NTCIR-13_MedWeb_{language_code}_test.xlsx",
                ),
            }
        elif self.config.schema == "bigbio_t2t":
            pattern = r"ntcir_13_medweb_translation_(?P<source_language_code>ja|en|zh)_(?P<target_language_code>ja|en|zh)_bigbio_t2t"
            match = re.search(pattern=pattern, string=self.config.subset_id)

            if not match:
                raise ValueError(
                    "Unable to parse source and target language codes for translation "
                    f"from dataset subset id: '{self.config.subset_id}'. Attempted to parse "
                    f"using this regex pattern: '{pattern}' but failed to get a match."
                )

            source_language_code = match.group("source_language_code")
            target_language_code = match.group("target_language_code")

            filepaths = {
                datasets.Split.TRAIN: (
                    Path(data_dir)
                    / f"NTCIR-13_MedWeb_{source_language_code}_training.xlsx",
                    Path(data_dir)
                    / f"NTCIR-13_MedWeb_{target_language_code}_training.xlsx",
                ),
                datasets.Split.TEST: (
                    Path(data_dir)
                    / f"NTCIR-13_MedWeb_{source_language_code}_test.xlsx",
                    Path(data_dir)
                    / f"NTCIR-13_MedWeb_{target_language_code}_test.xlsx",
                ),
            }

        return [
            datasets.SplitGenerator(
                name=split_name,
                gen_kwargs={
                    "filepaths": filepaths[split_name],
                    "split": split_name,
                },
            )
            for split_name in (datasets.Split.TRAIN, datasets.Split.TEST)
        ]

    def _language_from_filepath(self, filepath: Path):
        pattern = r"NTCIR-13_MedWeb_(?P<language_code>ja|en|zh)_(training|test).xlsx"
        match = re.search(pattern=pattern, string=filepath.name)

        if not match:
            raise ValueError(
                "Unable to parse language code from filename. "
                f"Filename was: '{filepath.name}' and tried to parse using this "
                f"regex pattern: '{pattern}' but failed to get a match."
            )

        return match.group("language_code")

    def _generate_examples(
        self, filepaths: Tuple[Path], split: str
    ) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        if self.config.schema == "source":
            dataframes = []

            for filepath in filepaths:
                language_code = self._language_from_filepath(filepath)
                df = pd.read_excel(filepath, sheet_name=f"{language_code}_{split}")
                df["Language"] = language_code
                dataframes.append(df)

            df = pd.concat(dataframes)

            for row_index, row in enumerate(df.itertuples(index=False)):
                yield row_index, row._asdict()

        elif self.config.schema == "bigbio_text":
            (filepath,) = filepaths
            language_code = self._language_from_filepath(filepath)

            df = pd.read_excel(
                filepath,
                sheet_name=f"{language_code}_{split}",
            )

            label_column_names = [
                column_name
                for column_name in df.columns
                if column_name not in ("ID", "Tweet")
            ]
            labels = (
                df[label_column_names]
                .apply(lambda row: row[row == "p"].index.tolist(), axis=1)
                .values
            )

            ids = df["ID"]
            tweets = df["Tweet"]

            for row_index, (record_labels, record_id, tweet) in enumerate(
                zip(labels, ids, tweets)
            ):
                yield row_index, {
                    "id": record_id,
                    "text": tweets,
                    "document_id": filepath.stem,
                    "labels": record_labels,
                }
        elif self.config.schema == "bigbio_t2t":
            source_filepath, target_filepath = filepaths

            source_language_code = self._language_from_filepath(source_filepath)
            target_language_code = self._language_from_filepath(target_filepath)

            source_df = pd.read_excel(
                source_filepath,
                sheet_name=f"{source_language_code}_{split}",
            )[["ID", "Tweet"]]
            source_df["id_int"] = source_df["ID"].str.extract(r"(\d+)").astype(int)

            target_df = pd.read_excel(
                target_filepath,
                sheet_name=f"{target_language_code}_{split}",
            )[["ID", "Tweet"]]
            target_df["id_int"] = target_df["ID"].str.extract(r"(\d+)").astype(int)

            df_combined = source_df.merge(
                target_df, on="id_int", suffixes=("_source", "_target")
            )[["id_int", "Tweet_source", "Tweet_target"]]

            for row_index, record in enumerate(df_combined.itertuples(index=False)):
                row = record._asdict()
                yield row_index, {
                    "id": f"{row['id_int']}_{source_language_code}_{target_language_code}",
                    "document_id": f"t2t_{source_language_code}_{target_language_code}",
                    "text_1": row["Tweet_source"],
                    "text_2": row["Tweet_target"],
                    "text_1_name": source_language_code,
                    "text_2_name": target_language_code,
                }
