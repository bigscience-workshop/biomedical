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
This template serves as a starting point for contributing a dataset to the BigScience Biomedical repo.

When modifying it for your dataset, look for TODO items that offer specific instructions.

Full documentation on writing dataset loading scripts can be found here:
https://huggingface.co/docs/datasets/add_dataset.html

To create a dataset loading script you will create a class and implement 3 methods:
  * `_info`: Establishes the schema for the dataset, and returns a datasets.DatasetInfo object.
  * `_split_generators`: Downloads and extracts data for each split (e.g. train/val/test) or associate local data with each split.
  * `_generate_examples`: Creates examples from data on disk that conform to each schema defined in `_info`.

TODO: Before submitting your script, delete this doc string and replace it with a description of your dataset.


[bigbio_schema_name] = (kb, pairs, qa, text, t2t, entailment)
"""

import os
import itertools as it
from typing import List, Tuple, Dict

import datasets
from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Tasks

_LOCAL = False

_CITATION = """\
@article{cho2022plant,
  author    = {Cho, Hyejin and Kim, Baeksoo and Choi, Wonjun and Lee, Doheon and Lee, Hyunju},
  title     = {Plant phenotype relationship corpus for biomedical relationships between plants and phenotypes},
  journal   = {Scientific Data},
  volume    = {9},
  year      = {2022},
  publisher = {Nature Publishing Group},
  doi       = {https://doi.org/10.1038/s41597-022-01350-1},
}
"""

_DATASETNAME = "plant_phenotype"

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\


Corpus: 

Annotators:

Annotation Quality:
"""

_HOMEPAGE = "https://github.com/DMCB-GIST/PPRcorpus"

_LICENSE = ""

# TODO: Add links to the urls needed to download your dataset files.
#  For local datasets, this variable can be an empty dictionary.

# For publicly available datasets you will most likely end up passing these URLs to dl_manager in _split_generators.
# In most cases the URLs will be the same for the source and bigbio config.
# However, if you need to access different files for each config you can have multiple entries in this dict.
# This can be an arbitrarily nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    _DATASETNAME: [
        "https://github.com/DMCB-GIST/PPRcorpus/blob/main/corpus/PPR_train_corpus.txt",
        "https://github.com/DMCB-GIST/PPRcorpus/blob/main/corpus/PPR_dev_corpus.txt",
        "https://github.com/DMCB-GIST/PPRcorpus/blob/main/corpus/PPR_test_corpus.txt",  
    ],
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]


_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"



class PlantPhenotypeDataset(datasets.GeneratorBasedBuilder):
    """\
    Plant-Phenotype is dataset for named-entity recognition and relation extraction of plants and their induced phenotypes
    """

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="plant_phenotype_source",
            version=SOURCE_VERSION,
            description="Plant Phenotype source schema",
            schema="source",
            subset_id="plant_phenotype",
        ),
        BigBioConfig(
            name="plant_phenotype_bigbio_kb",
            version=BIGBIO_VERSION,
            description="Plant Phenotype BigBio schema",
            schema="bigbio_kb",
            subset_id="plant_phenotype",
        ),
    ]

    DEFAULT_CONFIG_NAME = "plant_phenotype_source"

    def _info(self) -> datasets.DatasetInfo:

        # Create the source schema; this schema will keep all keys/information/labels as close to the original dataset as possible.

        # You can arbitrarily nest lists and dictionaries.
        # For iterables, use lists over tuples or `datasets.Sequence`

        if self.config.schema == "source":

            features = datasets.Features(
               {
                   "passage_id": datasets.Value("string"),
                   "pmid": datasets.Value("string"),
                   "section": datasets.Value("int32"),
                   "text": datasets.Value("string"), 
                #    "passages": [
                #        {
                #            'id': datasets.Value("string"),
                #            'text': datasets.Value("string"),
                #            'offsets': datasets.sequence(datasets.Value("int32")),
                #        }
                #    ]

                   "entities": [
                       {
                           "offsets": datasets.sequence(datasets.Value("int32")),
                           "text": datasets.Value("string"),
                           "type": datasets.Value("string"),
                        #    "entity_id": datasets.Value("string"),
                       }
                   ],
                   "relations": [
                       {
                           "relation_type": datasets.Value("string"),
                           "entity1_offsets": datasets.Sequence(datasets.Value("int32")),
                           "entity1_text": datasets.Value("string"),
                           "entity1_type": datasets.Value("string"),
                           "entity2_offsets": datasets.Sequence(datasets.Value("int32")),
                           "entity2_text": datasets.Value("string"),
                           "entity2_type": datasets.Value("string"),
                       }
                   ]
               }
            )

        elif self.config.schema == "bigbio_kb":
            features = schemas.kb_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        
        urls = _URLS[_DATASETNAME]
        train, dev, test = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # Whatever you put in gen_kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": train,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": test,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": dev,
                },
            ),
        ]


    def _generate_examples(self, filepath,) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        with open(filepath, 'r') as f:
            chunks = f.read().strip().split('\n\n')

        
        #EDIT
        if self.config.schema == 'source':
            for id_, doc in self._generate_source_examples(chunks):
                yield id_, doc
                    
            
        #EDIT
        elif self.config.schema == 'bigbio_kb':
            for id_, doc in self._generate_bigbio_kb_examples(chunks):
                yield id_, doc

    def _generate_whole_documents(self, chunks):
        '''
        Collect individual sentence annotations into whole abstracts
        '''
        prev_pmid = -1
        pmid = ""
        doc_chunks = []
        
        for c in chunks:
            lines = c.split('\n')
            dataset_id, passage_text = lines[0].split('\t')
            annotations = [l.split('\t') for l in lines[1:]]
            if len(annotations) == 0:
                continue
                
            
            # Get info on passage
            pmid, section = dataset_id.split('_')
            if prev_pmid == -1:
                prev_pmid = pmid
            if prev_pmid != pmid:
                out = {
                    'pmid': prev_pmid,
                    'doc_chunks': doc_chunks,
                }
                # Reset everything for next PMID
                prev_pmid = pmid
                
                yield out
                
                
                
            doc_chunks.append({'passage':passage_text, 'annotations':annotations, "sentence_id": dataset_id})
        
        # Take care of last document
        yield {
                'pmid': pmid,
                'doc_chunks': doc_chunks,
            }
    
    def _generate_source_examples(self, chunks):
        '''
        Generate examples in format of source schema
        '''
        
        for c in chunks:
            lines = c.split('\n')
            passage_id, passage_text = lines[0].split('\t')
            annotations = [l.split('\t') for l in lines[1:]]
            if len(annotations) == 0:
                continue

            # Get info on passage
            pmid, section = passage_id.split('_')
            section = int(section)
            pmid = annotations[0][0]

            # Grab entities and relations
            entities = []
            relations = []
            for a in annotations:
                if len(a) == 5:
                    entities.append({
                        "offsets": (int(a[1]), int(a[2])),
                        "text": a[3],
                        "type": a[4],
                    })

                elif len(a) == 10:
                    relations.append(
                        {
                        "relation_type": a[1],
                        "entity1_offsets": (int(a[2]),int(a[3])),
                        "entity1_text": a[4],
                        "entity1_type": a[5],
                        "entity2_offsets": (int(a[6]), int(a[7])),
                        "entity2_text": a[8],
                        "entity2_type": a[9],
                    }
                    )
                else:
                    # This is a special case that occurs for a single data point
                    relations.append(
                        {
                        "relation_type": a[1],
                        "entity1_offsets": (int(a[2]),int(a[3])),
                        "entity1_text": a[4],
                        "entity1_type": a[5],
                        "entity2_offsets": (int(a[8]), int(a[9])),
                        "entity2_text": a[10],
                        "entity2_type": a[11],
                    }
                    )
                    
                # Consolidate into document
                document = {
                    'passage_id': passage_id,
                    'pmid': pmid,
                    'section': section,
                    'text': passage_text,
                    'entities': entities,
                    'relations': relations,
                }
                
                yield passage_id, document

    def _generate_bigbio_kb_examples(self, chunks):
        '''
        Generator for training examples in bigbio_kb schema format
        '''
        uid = it.count(1)
        for document in self._generate_whole_documents(chunks):
            pmid = document['pmid']
            offset_delta = 0
            id_ = str(next(uid))
            
            passages = []
            entities = []
            relations = []

            # Iterate through each section of the article
            for c in document['doc_chunks']:
                passage = c['passage']
                annotations = c['annotations']

                passages.append(
                    {
                        "id": str(next(uid)),
                        "text": [passage],
                        "offsets": [(offset_delta, offset_delta + len(passage) )],
                    }
                )
                

                entities_sublist = []
                for a in annotations:
                    if len(a) == 5:
                        entities_sublist.append({
                            "id": str(next(uid)),
                            "type": a[4],
                            "text": [a[3]],
                            "offsets": [(int(a[1]) + offset_delta, int(a[2]) + offset_delta)],
                        })
                    
                        
                        
                # Create mapping of offsets to entity_id
                ent2id = {tuple(x['offsets']): x['id'] for x in entities_sublist}
                
                for a in annotations:
                    if len(a) == 10:
                        e1_offsets = [(int(a[2]) + offset_delta, int(a[3]) + offset_delta)]
                        e2_offsets = [(int(a[6]) + offset_delta, int(a[7]) + offset_delta)]
                        relations.append(
                            {
                                "id": str(next(uid)),
                                "type": a[1],
                                'arg1_id': ent2id[tuple(e1_offsets)],
                                'arg2_id': ent2id[tuple(e2_offsets)]
                        }
                        )


                    # Special case for a single annotation    
                    elif len(a) > 10:
                        e1_offsets = [(int(a[2]) + offset_delta, int(a[3]) + offset_delta)]
                        e2_offsets = [(int(a[8]) + offset_delta, int(a[9]) + offset_delta)]
                        relations.append(
                            {
                                "id": str(next(uid)),
                                "type": a[1],
                                'arg1_id': ent2id[tuple(e1_offsets)],
                                'arg2_id': ent2id[tuple(e2_offsets)]
                        }
                        )

                
                    
                offset_delta += len(passage) + 1

                
            doc = {
                    'id': id_,
                    'document_id': pmid,
                    'passages': passages,
                    'entities': entities,
                    'relations': relations,
                }
            
            yield id_, doc
            
            passages = []
            entities = []
            relations = []
            id_ = next(uid)
    


# This template is based on the following template from the datasets package:
# https://github.com/huggingface/datasets/blob/master/templates/new_dataset_script.py


# This allows you to run your dataloader with `python [dataset_name].py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    datasets.load_dataset(__file__)
