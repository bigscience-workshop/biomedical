from pathlib import Path
from typing import List

import datasets
import pandas as pd

from bigbio.utils import parsing, schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks

_DATASETNAME = "gad"
_SOURCE_VIEW_NAME = "source"
_UNIFIED_VIEW_NAME = "bigbio"

_LANGUAGES = [Lang.EN]
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

_HOMEPAGE = "https://github.com/dmis-lab/biobert" # This data source is used by the BLURB benchmark

_LICENSE = "Creative Common Attribution 4.0 International"

_URLs = {
    "source": "https://drive.google.com/uc?export=download&id=1-jDKGcXREb2X9xTFnuiJ36PvsqoyHWcw",
    "bigbio_text": "https://drive.google.com/uc?export=download&id=1-jDKGcXREb2X9xTFnuiJ36PvsqoyHWcw"
}

_SUPPORTED_TASKS = [
    Tasks.TEXT_CLASSIFICATION
]

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
        ) for i in range(10)
    ] + [
        # 10-fold bigbio schema
        BigBioConfig(
            name=f"gad_fold{i}_bigbio_text",
            version=datasets.Version(_BIGBIO_VERSION),
            description="GAD BigBio schema",
            schema="bigbio_text",
            subset_id=f"gad_fold{i}",
        ) for i in range(10)
    ]

    DEFAULT_CONFIG_NAME = "gad_fold0_source"

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "index": datasets.Value("string"),
                    "sentence": datasets.Value("string"),
                    "label": datasets.Value("int32")
                }
            )
        elif self.config.schema == "bigbio_text":
            features = schemas.text_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:
        fold_id = int(self.config.subset_id.split("_fold")[1][0]) + 1
        
        my_urls = _URLs[self.config.schema]
        data_dir = Path(dl_manager.download_and_extract(my_urls))
        data_files = {
            "train": data_dir / "GAD" / str(fold_id) / "train.tsv",
            "test": data_dir / "GAD" / str(fold_id) / "test.tsv"
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
        if 'train.tsv' in str(filepath):
            df = pd.read_csv(filepath, sep='\t', header=None).reset_index()
        else:
            df = pd.read_csv(filepath, sep='\t')
        df.columns = ['id', 'sentence', 'label']

        if self.config.schema == "source":
            for id, row in enumerate(df.itertuples()):
                ex = {
                    "index": row.id,
                    "sentence": row.sentence,
                    "label": int(row.label)
                }                
                yield id, ex
        elif self.config.schema == "bigbio_text":
            for id, row in enumerate(df.itertuples()):
                ex = {
                    "id": id,
                    "document_id": row.id,
                    "text": row.sentence,
                    "labels": [str(row.label)]
                }
                yield id, ex            
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
