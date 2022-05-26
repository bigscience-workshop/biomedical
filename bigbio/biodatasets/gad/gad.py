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
@inproceedings{10.5555/2107691.2107703,
author = {Pyysalo, Sampo and Ohta, Tomoko and Tsujii, Jun'ichi},
title = {Overview of the Entity Relations (REL) Supporting Task of BioNLP Shared Task 2011},
year = {2011},
isbn = {9781937284091},
publisher = {Association for Computational Linguistics},
address = {USA},
abstract = {This paper presents the Entity Relations (REL) task,
a supporting task of the BioNLP Shared Task 2011. The task concerns
the extraction of two types of part-of relations between a gene/protein
and an associated entity. Four teams submitted final results for
the REL task, with the highest-performing system achieving 57.7%
F-score. While experiments suggest use of the data can help improve
event extraction performance, the task data has so far received only
limited use in support of event extraction. The REL task continues
as an open challenge, with all resources available from the shared
task website.},
booktitle = {Proceedings of the BioNLP Shared Task 2011 Workshop},
pages = {83–88},
numpages = {6},
location = {Portland, Oregon},
series = {BioNLP Shared Task '11}
}
"""

_DESCRIPTION = """\
The Entity Relations (REL) task is a supporting task of the BioNLP Shared Task 2011.
The task concerns the extraction of two types of part-of relations between a
gene/protein and an associated entity.
"""

_HOMEPAGE = "https://github.com/openbiocorpora/bionlp-st-2011-rel"

_LICENSE = "License Terms for BioNLP Shared Task 2011 Data"

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
        """
        - `features` defines the schema of the parsed data set. The schema depends on the
        chosen `config`: If it is `_SOURCE_VIEW_NAME` the schema is the schema of the
        original data. If `config` is `_UNIFIED_VIEW_NAME`, then the schema is the
        canonical KB-task schema defined in `biomedical/schemas/kb.py`.
        """
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
