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

import os
from typing import List, Tuple, Dict

import datasets
from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

_CITATION = """\
@article{Caporaso2007,
author = {J. Gregory Caporaso and William A Baumgartner and David A Randolph and K. Bretonnel Cohen and Lawrence Hunter},
title = {MutationFinder: a high-performance system for extracting point mutation mentions from text.},
journal = {Bioinformatics},
year = {2007},
volume = {23},
pages = {1862--1865},
number = {14},
month = {Jul},
pii = {btm235},
pmid = {17495998},
timestamp = {2013.01.15},
url = {http://dx.doi.org/10.1093/bioinformatics/btm235}
}
"""

_DATASETNAME = "mutation_finder"

_DESCRIPTION = """\
Gold standard corpus for mutation extraction systems consisting of 1515 human-annotated mutation mentions in 813 MEDLINE
abstracts. This corpus is divided into development and test subsets. Interannotator agreement on this corpus, judged on 
fifty abstracts, was 94%. 
"""

_HOMEPAGE = "http://mutationfinder.sourceforge.net/"

_LICENSE = """\
Copyright (c) 2007 Regents of the University of Colorado

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
"""

_URLS = {
    _DATASETNAME: "https://sourceforge.net/projects/mutationfinder/files/MutationFinder/MutationFinder-1.1/MutationFinder-1.1.tar.gz/download",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "1.1.0"

_BIGBIO_VERSION = "1.0.0"


class MutationFinderDataset(datasets.GeneratorBasedBuilder):
    """Gold standard mutation corpus released alongside the MutationFinder mutation extraction system."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="mutation_finder_source",
            version=SOURCE_VERSION,
            description="mutation_finder source schema",
            schema="source",
            subset_id="mutation_finder",
        ),
        BigBioConfig(
            name="mutation_finder_bigbio_kb",
            version=BIGBIO_VERSION,
            description="mutation_finder BigBio schema",
            schema="bigbio_kb",
            subset_id="mutation_finder",
        ),
    ]

    DEFAULT_CONFIG_NAME = "mutation_finder_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
               {
                   "identifier": datasets.Value("string"),
                   "text": datasets.Value("string"),
                   "mutations": [
                       {
                           "text": datasets.Value("string"),
                       }
                   ],
               }
            )
        elif self.config.schema == "bigbio_kb":
            features = schemas.kb_features
        else:
            raise NotImplementedError(self.config.schema)

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "doc_collection_path": os.path.join(data_dir, "MutationFinder/corpora/devo_set.txt"),
                    "gold_std_path": os.path.join(data_dir, "MutationFinder/corpora/devo_gold_std.txt"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "doc_collection_path": os.path.join(data_dir, "MutationFinder/corpora/test_set.txt"),
                    "gold_std_path": os.path.join(data_dir, "MutationFinder/corpora/test_gold_std.txt"),
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, doc_collection_path, gold_std_path, split: str) -> Tuple[int, Dict]:
        with open(doc_collection_path) as doc_collection, open(gold_std_path) as gold_std:
            # First, parse doc_id -> mutations from the gold standard
            # Required, because lines in the gold standard and doc collection file are out of order.
            mutations = {}
            for line in gold_std:
                values = line.strip().split('\t')
                identifier = values[0]
                mutations[identifier] = values[1:] if len(values) > 1 else []

            uid = 0
            for idx, line in enumerate(doc_collection):
                values = line.strip().split('\t')
                identifier = values[0]

                if self.config.schema == "source":
                    # remove null values from the text
                    text = '\t'.join(v for v in values[1:] if v != 'null')

                    yield idx, {
                        "identifier": identifier,
                        "text": text,
                        "mutations": [{'text': m} for m in mutations[identifier]]
                    }
                elif self.config.schema == "bigbio_kb":
                    values = line.strip().split('\t')
                    identifier = values[0]

                    passages = []
                    for p_text in values[1:]:
                        if p_text != 'null':
                            passages.append(
                                {
                                    "id": uid,
                                    "type": "",
                                    "text": [p_text],
                                    "offsets": [],
                                }
                            )
                            uid += 1

                    entities = []
                    for m_text in mutations[identifier]:
                        entities.append(
                            {
                                "id": uid,
                                "type": "",
                                "text": [m_text],
                                "offsets": [],
                                "normalized": []
                            }
                        )
                        uid += 1

                    yield idx, {
                        "id": uid,
                        "document_id": identifier,
                        "passages": passages,
                        "entities": entities,
                        "events": [],
                        "coreferences": [],
                        "relations": []
                    }
                    uid += 1
                else:
                    raise NotImplementedError(self.config.schema)


# This allows you to run your dataloader with `python [dataset_name].py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    dset = datasets.load_dataset(__file__, name='mutation_finder_bigbio_kb')
