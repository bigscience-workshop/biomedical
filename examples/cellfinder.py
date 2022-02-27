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

"""

"""
#  < Your imports here >
import os  # useful for paths
from typing import Dict, Iterable, List

import datasets
import pybrat
from pybrat.parser import BratParser


_CITATION = """\
@article{,
  author    = {},
  title     = {},
  journal   = {},
  volume    = {},
  year      = {},
  url       = {},
  doi       = {},
  biburl    = {},
  bibsource = {}
}
"""
_DATASETNAME = "CellFinder"

_DESCRIPTION = """CellFinder dataset  aims to create a stem cell data repository by linking information from existing 
public databases and by performing text mining on the research literature. It is composed of 10 full text documents 
containing more than 2,100 sentences, 65,000 tokens and 5,200 annotations for entities. The corpus has been annotated 
with six types of entities (anatomical parts, cell components, cell lines, cell types, genes/protein and species) with 
an overall inter-annotator agreement around 80%. The dataset is in Brat format"""

_HOMEPAGE = "https://www.informatik.hu-berlin.de/de/forschung/gebiete/wbi/resources/cellfinder/"

_LICENSE = ""

_URLs = {"source": "https://www.informatik.hu-berlin.de/de/forschung/gebiete/wbi/resources/cellfinder/cellfinder1_brat.tar.gz",
         "bigbio": "https://www.informatik.hu-berlin.de/de/forschung/gebiete/wbi/resources/cellfinder/cellfinder1_brat.tar.gz" }

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"

class CellFinderDataset(datasets.GeneratorBasedBuilder):
    """Write a short docstring documenting what this dataset is"""

    VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="source",
            version=VERSION,
            description="Source schema"
        ),
        datasets.BuilderConfig(
            name="bigbio",
            version=BIGBIO_VERSION,
            description="BigScience Biomedical schema",
        ),
    ]

    DEFAULT_CONFIG_NAME = "source"

    def _info(self):

        if self.config.name == "source":
            features = datasets.Features(
                {
                    "article_id": datasets.Value("int32"),
                    "text": datasets.Value("string"),
                    "entities": datasets.Sequence(
                        {
                            "spans": datasets.Sequence(datasets.Value("int32")),
                            "text": datasets.Value("string"),
                            "entity_type": datasets.Value("string"),
                        }
                    ),
                    "relations": datasets.Sequence(
                        {
                            "relation_type": datasets.Value("string"),
                            "arg1": datasets.Value("string"),
                            "arg2": datasets.Value("string"),
                        }
                    ),
                }
            )

        if self.config.name == "bigbio":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "passages": [
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "text": datasets.Sequence(datasets.Value("string")),
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                        }
                    ],
                    "entities": [
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "text": datasets.Sequence(datasets.Value("string")),
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                            "normalized": [
                                {
                                    "db_name": datasets.Value("string"),
                                    "db_id": datasets.Value("string"),
                                }
                            ],
                        }
                    ],

                    "relations": [
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "arg1_id": datasets.Value("string"),
                            "arg2_id": datasets.Value("string"),
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

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        # No need for a download location; just specify file location
        data_dir = os.getcwd()
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir,
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):

        if self.config.name == "source":
            datadir = os.path.join(filepath, split)
            parser = BratParser(error="ignore")
            data = self._parse_brat_source(datadir, parser)
            for id_, key in enumerate(data):
                yield id_, key

        elif self.config.name == "bigbio":
            datadir = os.path.join(filepath, split)
            parser = BratParser(error="ignore")
            data = self._parse_brat_bigbio(datadir, parser)
            for id_, key in enumerate(data):
                yield id_, key

    @staticmethod
    def _parse_brat_source(data_dir: str, parser: pybrat.parser.BratParser) -> List[Dict[str, object]]:
        """ """
        # Format is Example[entities, relations]

        data = parser.parse(data_dir)
        output = []

        for x in data:
            pmid = x.id
            text = x.text

            ents = []
            relns = []

            for e in x.entities:
                ents.append(
                    {
                        "spans": [e.start, e.end],
                        "text": e.type,
                        "entity_type": e.mention,
                    }
                )

            for r in x.relations:
                if len(r):
                    relns.append(
                        {
                            "relation_type": None,
                            "arg1": None,
                            "arg2": None,
                        }
                    )
                else:
                    relns.append(
                        {
                            "relation_type": None,
                            "arg1": None,
                            "arg2": None,
                        }
                    )

            output.append(
                {
                    "article_id": pmid,
                    "text": text,
                    "entities": ents,
                    "relations": relns,
                }
            )
        return output

    @staticmethod
    def _parse_brat_bigbio(data_dir: str, parser: pybrat.parser.BratParser) -> List[Dict[str, object]]:
        """
        # Format is Example[entities, relations]
        TODO: implement Bigbio schema here

        features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "passages": [
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "text": datasets.Sequence(datasets.Value("string")),
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                        }
                    ],
                    "entities": [
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "text": datasets.Sequence(datasets.Value("string")),
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                            "normalized": [
                                {
                                    "db_name": datasets.Value("string"),
                                    "db_id": datasets.Value("string"),
                                }
                            ],
                        }
                    ],

                    "relations": [
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "arg1_id": datasets.Value("string"),
                            "arg2_id": datasets.Value("string"),
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



        """
        # data = parser.parse(data_dir)
        # output = []
        #
        # for x in data:
        #     pmid = x.id
        #     text = x.text
        #
        #     ents = []
        #     relns = []
        #
        #     for e in x.entities:
        #         ents.append(
        #             {
        #                 "spans": [e.start, e.end],
        #                 "text": e.type,
        #                 "entity_type": e.mention,
        #             }
        #         )
        #
        #     for r in x.relations:
        #         if len(r):
        #             relns.append(
        #                 {
        #                     "relation_type": None,
        #                     "arg1": None,
        #                     "arg2": None,
        #                 }
        #             )
        #         else:
        #             relns.append(
        #                 {
        #                     "relation_type": None,
        #                     "arg1": None,
        #                     "arg2": None,
        #                 }
        #             )
        #
        #     output.append(
        #         {
        #             "article_id": pmid,
        #             "text": text,
        #             "entities": ents,
        #             "relations": relns,
        #         }
        #     )
        # return output
        pass


if __name__ == "__main__":
    from datasets import load_dataset
    dataset = load_dataset(__file__, name= "source")
    print(dataset)