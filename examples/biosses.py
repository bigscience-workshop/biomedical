
   
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
BioSSES computes similarity of biomedical sentences by utilizing WordNet as the general domain  ontology
and UMLS as the biomedical domain specific ontology.
The original paper outlines the approaches with respect to using annotator score as golden standard.
Source view will return all annotator score individually whereas the Bigbio view will return the mean of the annotator score
"""
import datasets
from datasets import load_dataset
import pandas as pd


_DATASETNAME = "biosses"

_CITATION = """\
@article{souganciouglu2017biosses,
  author={Soğancıoğlu, Gizem, Hakime Öztürk, and Arzucan Özgür},
  title = {BIOSSES: a semantic sentence similarity estimation system for the biomedical domain},
  journal={Bioinformatics},
  volume={33},
  number={14},
  pages={i49--i58},
  year={2017},
  publisher={Oxford University Press}
}
"""

_DESCRIPTION = """
BioSSES computes similarity of biomedical sentences by utilizing WordNet as the general domain  ontology 
and UMLS as the biomedical domain specific ontology.
The original paper outlines the approaches with respect to using annotator score as golden standard. 
Source view will return all annotator score individually whereas the Bigbio view will return the mean of the annotator score
"""

_HOMEPAGE = "https://tabilab.cmpe.boun.edu.tr/BIOSSES/SourceCode.html"

_LICENSE = ""


_URLs = {
    "source": "https://raw.githubusercontent.com/debajyotidatta/biomedical/biosses_add/examples/biosses/biosses_full.tsv",
    "bigbio": "https://raw.githubusercontent.com/debajyotidatta/biomedical/biosses_add/examples/biosses/biosses_full.tsv",
}

_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class Biosses(datasets.GeneratorBasedBuilder):
    """BIOSSES : Biomedical Semantic Similarity Estimation System"""

    VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)


    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="source",
            version=VERSION,
            description=_DESCRIPTION,
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
                    "id": datasets.Value("int64"),
                    "document_id": datasets.Value("int64"),
                    "text_1": datasets.Value("string"),
                    "text_2": datasets.Value("string"),
                    "annotator_a": datasets.Value("int64"),
                    "annotator_b": datasets.Value("int64"),
                    "annotator_c": datasets.Value("int64"),
                    "annotator_d": datasets.Value("int64"),
                    "annotator_e": datasets.Value("int64")
                }
            )
        elif self.config.name == "bigbio":
            features = datasets.Features(
                {   "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "text_1": datasets.Value("string"),
                    "text_2": datasets.Value("string"),
                    "label": datasets.Value("string"),
                }
            )


        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        my_urls = _URLs[self.config.name]
        dl_dir = dl_manager.download_and_extract(my_urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": dl_dir,
                    "split": "train",
                }
            )
        ]

    def _generate_examples(self, filepath,  split):

        df = pd.read_csv(filepath, sep='\t', encoding='utf-8')
        key = 1
        for id_, row in df.iterrows():
            print(row)
            if self.config.name == "source":
                yield id_, {
                    "id": key,
                    "document_id": row['sentence_id'],
                    "text_1": row['sentence_1'],
                    "text_2": row['sentence_2'],
                    "annotator_a": row['annotator_a'],
                    "annotator_b": row['annotator_b'],
                    "annotator_c": row['annotator_c'],
                    "annotator_d": row['annotator_d'],
                    "annotator_e": row['annotator_e']
                }
            if self.config.name == "bigbio":
                yield id_, {
                    "id": key,
                    "document_id": row['sentence_id'],
                    "text_1": row['sentence_1'],
                    "text_2": row['sentence_2'],
                    "label": str((row['annotator_a']+row['annotator_b']+ row['annotator_c']+row['annotator_d']+ row['annotator_e'])/5)

                }
            key += 1

if __name__ == "__main__":
    from datasets import load_dataset
    ds = load_dataset(__file__)
    print(ds)