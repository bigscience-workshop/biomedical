import datasets
import os

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

_DatasetName = "euadr"

_DESCRIPTION = """\
Corpora with specific entities and relationships annotated are essential to train and evaluate text-mining systems that are developed to extract specific structured information from a large corpus. In this paper we describe an approach where a named-entity recognition system produces a first annotation and annotators revise this annotation using a web-based interface. The agreement figures achieved show that the inter-annotator agreement is much better than the agreement with the system provided annotations. The corpus has been annotated for drugs, disorders, genes and their inter-relationships. For each of the drug–disorder, drug–target, and target–disorder relations three experts have annotated a set of 100 abstracts. These annotated relationships will be used to train and evaluate text-mining software to capture these relationships in texts.

"""

_HOMEPAGE = "https://www.sciencedirect.com/science/article/pii/S1532046412000573"

_LICENSE = "Elsevier user license"

_URL = "https://biosemantics.erasmusmc.nl/downloads/euadr.tgz"

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
                    "pmid": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "abstract": datasets.Value("string"),
                    "annotations": datasets.Sequence(datasets.Value("string")),
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
        datapath = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"datapath": datapath, "dl_manager": dl_manager},
            ),
        ]

    def _generate_examples(self, datapath, dl_manager):
        def replace_html_special_chars(string):
            # since we are getting the text as an HTML file, we need to replace
            # special characters
            for (i, r) in [
                ("&#34;", '"'),
                ("&quot;", '"'),
                ("&#39;", "'"),
                ("&apos;", "'"),
                ("&#38;", "&"),
                ("&amp;", "&"),
                ("&#60;", "<"),
                ("&lt;", "<"),
                ("&#62;", ">"),
                ("&gt;", ">"),
                ("&#x27;", "'"),
            ]:
                string = string.replace(i, r)
            return string

        def suppr_blank(l_str):
            r = []
            for string in l_str:
                if len(string) > 0:
                    r.append(string)
            return r

        folder_path = os.path.join(datapath, "euadr_corpus")
        key = 0
        if self.config.schema == "source":
            for filename in os.listdir(folder_path):
                if "_" not in filename:
                    corpus_path = dl_manager.download_and_extract(
                        f"https://pubmed.ncbi.nlm.nih.gov/{filename[:-4]}/?format=pubmed"
                    )
                    with open(corpus_path, "r", encoding="latin") as f:
                        full_html = replace_html_special_chars(
                            ("".join(f.readlines()))
                            .replace("\r\n", "")
                            .replace("\n", "")
                        )
                        abstract = " ".join(
                            suppr_blank(
                                full_html.split("AB  -")[-1]
                                .split("FAU -")[0]
                                .split(" ")
                            )
                        )
                        title = " ".join(
                            suppr_blank(
                                full_html.split("TI  -")[-1].split("PG")[0].split(" ")
                            )
                        )
                        full_text = " ".join([title, abstract])
                    with open(
                        os.path.join(folder_path, filename), "r", encoding="latin"
                    ) as f:
                        lines = f.readlines()
                    yield key, {
                        "pmid": filename[:-4],
                        "title": title,
                        "abstract": abstract,
                        "annotations": lines,
                    }
                    key += 1
        elif self.config.schema == "bigbio_kb":
            for filename in os.listdir(folder_path):
                if "_" not in filename:
                    corpus_path = dl_manager.download_and_extract(
                        f"https://pubmed.ncbi.nlm.nih.gov/{filename[:-4]}/?format=pubmed"
                    )
                    with open(corpus_path, "r", encoding="latin") as f:
                        full_html = replace_html_special_chars(
                            ("".join(f.readlines()))
                            .replace("\r\n", "")
                            .replace("\n", "")
                        )
                        abstract = " ".join(
                            suppr_blank(
                                full_html.split("AB  -")[-1]
                                .split("FAU -")[0]
                                .split(" ")
                            )
                        )
                        title = " ".join(
                            suppr_blank(
                                full_html.split("TI  -")[-1].split("PG")[0].split(" ")
                            )
                        )
                        full_text = " ".join([title, abstract])
                    with open(
                        os.path.join(folder_path, filename), "r", encoding="latin"
                    ) as f:
                        lines = f.readlines()
                        data = {
                            "id": str(key),
                            "document_id": str(key),
                            "passages": [],
                            "entities": [],
                            "events": [],
                            "coreferences": [],
                            "relations": [],
                        }
                        key += 1
                        data["passages"].append(
                            {
                                "id": str(key),
                                "type": "title",
                                "text": [title],
                                "offsets": [[0, len(title)]],
                            }
                        )
                        key += 1
                        data["passages"].append(
                            {
                                "id": str(key),
                                "type": "abstract",
                                "text": [abstract],
                                "offsets": [
                                    [len(title) + 1, len(title) + 1 + len(abstract)]
                                ],
                            }
                        )
                        key += 1
                        for line in lines:
                            line_processed = line.split("\t")
                            if line_processed[2] == "relation":
                                data["entities"].append(
                                    {
                                        "id": str(key),
                                        "offsets": [
                                            [
                                                int(line_processed[7].split(":")[0]),
                                                int(line_processed[7].split(":")[1]),
                                            ]
                                        ],
                                        "text": [
                                            full_text[
                                                int(
                                                    line_processed[7].split(":")[0]
                                                ) : int(line_processed[7].split(":")[1])
                                            ]
                                        ],
                                        "type": "",
                                        "normalized": [
                                            {"db_name": None, "db_id": None}
                                        ],
                                    }
                                )
                                key += 1
                                data["entities"].append(
                                    {
                                        "id": str(key),
                                        "offsets": [
                                            [
                                                int(line_processed[8].split(":")[0]),
                                                int(line_processed[8].split(":")[1]),
                                            ]
                                        ],
                                        "text": [
                                            full_text[
                                                int(
                                                    line_processed[8].split(":")[0]
                                                ) : int(line_processed[8].split(":")[1])
                                            ]
                                        ],
                                        "type": "",
                                        "normalized": [
                                            {"db_name": None, "db_id": None}
                                        ],
                                    }
                                )
                                key += 1
                                data["relations"].append(
                                    {
                                        "id": str(key),
                                        "type": line_processed[-1].split("\n")[0],
                                        "arg1_id": str(key - 2),
                                        "arg2_id": str(key - 1),
                                        "normalized": [
                                            {"db_name": None, "db_id": None}
                                        ],
                                    }
                                )
                                key += 1
                            elif line_processed[2] == "concept":
                                data["entities"].append(
                                    {
                                        "id": str(key),
                                        "offsets": [
                                            [
                                                int(line_processed[4]),
                                                int(line_processed[5]),
                                            ]
                                        ],
                                        "text": [
                                            full_text[
                                                int(line_processed[4]) : int(
                                                    line_processed[5]
                                                )
                                            ]
                                        ],
                                        "type": line_processed[-1].split("\n")[0],
                                        "normalized": [
                                            {"db_name": None, "db_id": None}
                                        ],
                                    }
                                )
                                key += 1
                    yield key, data
                    key += 1
