import datasets
import os

import pandas as pd
import numpy as np

from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

_CITATION = """\
@article{VANMULLIGEN2012879,
title = {The EU-ADR corpus: Annotated drugs, diseases, targets, and their relationships},
journal = {Journal of Biomedical Informatics},
volume = {45},
number = {5},
pages = {879-884},
year = {2012},
note = {Text Mining and Natural Language Processing in Pharmacogenomics},
issn = {1532-0464},
doi = {https://doi.org/10.1016/j.jbi.2012.04.004},
url = {https://www.sciencedirect.com/science/article/pii/S1532046412000573},
author = {Erik M. {van Mulligen} and Annie Fourrier-Reglat and David Gurwitz and Mariam Molokhia and Ainhoa Nieto and Gianluca Trifiro and Jan A. Kors and Laura I. Furlong},
keywords = {Text mining, Corpus development, Machine learning, Adverse drug reactions},
abstract = {Corpora with specific entities and relationships annotated are essential to train and evaluate text-mining systems that are developed to extract specific structured information from a large corpus. In this paper we describe an approach where a named-entity recognition system produces a first annotation and annotators revise this annotation using a web-based interface. The agreement figures achieved show that the inter-annotator agreement is much better than the agreement with the system provided annotations. The corpus has been annotated for drugs, disorders, genes and their inter-relationships. For each of the drug–disorder, drug–target, and target–disorder relations three experts have annotated a set of 100 abstracts. These annotated relationships will be used to train and evaluate text-mining software to capture these relationships in texts.}
}
"""

_DatasetName="euadr"

_DESCRIPTION = """\
Corpora with specific entities and relationships annotated are essential to train and evaluate text-mining systems that are developed to extract specific structured information from a large corpus. In this paper we describe an approach where a named-entity recognition system produces a first annotation and annotators revise this annotation using a web-based interface. The agreement figures achieved show that the inter-annotator agreement is much better than the agreement with the system provided annotations. The corpus has been annotated for drugs, disorders, genes and their inter-relationships. For each of the drug–disorder, drug–target, and target–disorder relations three experts have annotated a set of 100 abstracts. These annotated relationships will be used to train and evaluate text-mining software to capture these relationships in texts.

"""

_HOMEPAGE = "https://www.sciencedirect.com/science/article/pii/S1532046412000573"

_LICENSE = "Elsevier user license"

_URL = "https://raw.githubusercontent.com/EsmaeilNourani/Deep-GDAE/master/data/EUADR_target_disease.csv"

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

class EUADR(datasets.GeneratorBasedBuilder):
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    DEFAULT_CONFIG_NAME = "euadr_bigbio_kb"

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="euadr_source",
            version=SOURCE_VERSION,
            description="EU-ADR source schema",
            schema="source",
            subset_id="euadr",
        ),
        BigBioConfig(
            name="euadr_bigbio_kb",
            version=BIGBIO_VERSION,
            description="EU-ADR simplified BigBio schema for named entity recognition and relation extraction",
            schema="bigbio_kb",
            subset_id="euadr",
        ),
    ]

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                                "ASSOCIATION_TYPE": datasets.Value("string"),
                                "PMID": datasets.Value("int32"),
                                "NUM_SENTENCE": datasets.Value("int32"),
                                "ENTITY1_TEXT": datasets.Value("string"),
                                "ENTITY1_INI": datasets.Value("int32"),
                                "ENTITY1_END": datasets.Value("int32"),
                                "ENTITY1_TYPE": datasets.Value("string"),
                                "ENTITY2_TEXT": datasets.Value("string"),
                                "ENTITY2_INI": datasets.Value("int32"),
                                "ENTITY2_END": datasets.Value("int32"),
                                "ENTITY2_TYPE": datasets.Value("string"),
                                "SENTENCE": datasets.Value("string"),
                }
            )
        elif self.config.schema == "bigbio_kb":
            features = schemas.kb_features
            
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )


    def _split_generators(self, dl_manager):
        urls = _URL
        data_file = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "datafile": data_file
                },
            ),
        ]

    def _generate_examples(self, datafile):
        key = 0
        filepath = os.path.join(datafile)
        datatable = pd.read_csv(filepath, sep='\t', encoding='latin', keep_default_na=False)
        if self.config.schema == "source":
            for i in range(datatable.shape[0]):
                data = {}
                for column in datatable.columns:
                    data[column] = datatable.loc[i, column]
                yield key, data
                key += 1
        elif self.config.schema == "bigbio_kb":
            for i in range(datatable.shape[0]):
                data = {
                    "id": str(key),
                    "document_id": str(key),
                    "passages": [],
                    "entities": [],
                    "relations": [],
                    "events": [],
                    "coreferences": [],
                }
                key += 1
                data["passages"].append({
                    "id": str(key),
                    "type": "sentence",
                    "text": [datatable.loc[i, "SENTENCE"]],
                    "offsets": [[0, len(datatable.loc[i, "SENTENCE"])]],
                    })
                key+=1
                for entity in ["ENTITY1", "ENTITY2"]:
                    data["entities"].append({
                        "id": str(key),
                        "offsets" : [[int(datatable.loc[i, f"{entity}_INI"]), int(datatable.loc[i, f"{entity}_END"])]],
                        "text": [datatable.loc[i, f"{entity}_TEXT"]],
                        "type": str(datatable.loc[i, f"{entity}_TYPE"]),
                        "normalized": [{"db_name": None, "db_id": None}]
                        })
                    key += 1
                data["relations"].append({
                    "id": str(key),
                    "type": str(datatable.loc[i, "ASSOCIATION_TYPE"]),
                    "arg1_id": str(key-2),
                    "arg2_id": str(key-1),
                    "normalized": [{"db_name": None, "db_id": None}],
                    })
                key+=1

                yield key, data
                