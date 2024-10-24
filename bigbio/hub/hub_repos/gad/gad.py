from pathlib import Path
from typing import List

import datasets
import pandas as pd

from .bigbiohub import text_features
from .bigbiohub import BigBioConfig
from .bigbiohub import Tasks


_SOURCE_VIEW_NAME = "source"
_UNIFIED_VIEW_NAME = "bigbio"

_LANGUAGES = ["English"]
_PUBMED = True
_LOCAL = False
_CITATION = """\
@article{Bravo2015,
  doi = {10.1186/s12859-015-0472-9},
  url = {https://doi.org/10.1186/s12859-015-0472-9},
  year = {2015},
  month = feb,
  publisher = {Springer Science and Business Media {LLC}},
  volume = {16},
  number = {1},
  author = {{\`{A}}lex Bravo and Janet Pi{\~{n}}ero and N{\'{u}}ria Queralt-Rosinach and Michael Rautschka and Laura I Furlong},
  title = {Extraction of relations between genes and diseases from text and large-scale data analysis: implications for translational research},
  journal = {{BMC} Bioinformatics}
}
"""

_DESCRIPTION = """\
A corpus identifying associations between genes and diseases by a semi-automatic
annotation procedure based on the Genetic Association Database
"""

_DATASETNAME = "gad"
_DISPLAYNAME = "GAD"

_HOMEPAGE = "https://github.com/dmis-lab/biobert"  # This data source is used by the BLURB benchmark

_LICENSE = "CC_BY_4p0"

_SUPPORTED_TASKS = [Tasks.TEXT_CLASSIFICATION]

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class GAD(datasets.GeneratorBasedBuilder):
    """GAD is a weakly labeled dataset for Entity Relations (REL) task which is treated as a sentence classification task."""

    BUILDER_CONFIGS = [
        # 10-fold source schema
        BigBioConfig(
            name=f"gad_fold{i}_source",
            version=datasets.Version(_SOURCE_VERSION),
            description="GAD source schema",
            schema="source",
            subset_id=f"gad_fold{i}",
        )
        for i in range(10)
    ] + [
        # 10-fold bigbio schema
        BigBioConfig(
            name=f"gad_fold{i}_bigbio_text",
            version=datasets.Version(_BIGBIO_VERSION),
            description="GAD BigBio schema",
            schema="bigbio_text",
            subset_id=f"gad_fold{i}",
        )
        for i in range(10)
    ]

    # BLURB Benchmark config https://microsoft.github.io/BLURB/
    BUILDER_CONFIGS.append(
        BigBioConfig(
            name=f"gad_blurb_bigbio_text",
            version=datasets.Version(_BIGBIO_VERSION),
            description=f"GAD BLURB benchmark in simplified BigBio schema",
            schema="bigbio_text",
            subset_id=f"gad_blurb",
        )
    )

    DEFAULT_CONFIG_NAME = "gad_fold0_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "index": datasets.Value("string"),
                    "sentence": datasets.Value("string"),
                    "label": datasets.Value("int32"),
                }
            )
        elif self.config.schema == "bigbio_text":
            features = text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:

        data_dir = Path(dl_manager.download_and_extract("data/REdata.zip"))

        if "blurb" in self.config.name:
            data_files = {
                "train": data_dir / "GAD" / "blurb" / "train.tsv",
                "validation": data_dir / "GAD" / "blurb" / "dev.tsv",
                "test": data_dir / "GAD" / "blurb" / "test.tsv",
            }

            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={"filepath": data_files["train"]},
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={"filepath": data_files["validation"]},
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={"filepath": data_files["test"]},
                ),
            ]

        else:

            fold_id = int(self.config.subset_id.split("_fold")[1][0]) + 1

            data_files = {
                "train": data_dir / "GAD" / str(fold_id) / "train.tsv",
                "test": data_dir / "GAD" / str(fold_id) / "test.tsv",
            }

            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={"filepath": data_files["train"]},
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={"filepath": data_files["test"]},
                ),
            ]

    def _generate_examples(self, filepath: Path):
        # train files in non-blurb splits don't have headers for some reason
        if "train.tsv" in str(filepath) and "blurb" not in self.config.name:
            df = pd.read_csv(filepath, sep="\t", header=None).reset_index()
        else:
            df = pd.read_csv(filepath, sep="\t")
        df.columns = ["id", "sentence", "label"]

        if self.config.schema == "source":
            for id, row in enumerate(df.itertuples()):
                ex = {
                    "index": row.id,
                    "sentence": row.sentence,
                    "label": int(row.label),
                }
                yield id, ex
        elif self.config.schema == "bigbio_text":
            for id, row in enumerate(df.itertuples()):
                ex = {
                    "id": id,
                    "document_id": row.id,
                    "text": row.sentence,
                    "labels": [str(row.label)],
                }
                yield id, ex
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
