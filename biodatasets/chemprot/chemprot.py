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
The BioCreative VI Chemical-Protein interaction dataset identifies entities of chemicals and proteins and their likely
relation to one other. Compounds are generally agonists (activators) or antagonists (inhibitors) of proteins.
The script loads dataset in bigbio schema (using knowledgebase schema: schemas/kb) AND/OR source (default) schema

"""
import os
from typing import Dict, Tuple

import datasets
from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Tasks

_LOCAL = False
_CITATION = """\
@article{DBLP:journals/biodb/LiSJSWLDMWL16,
  author    = {Krallinger, M., Rabal, O., Lourenço, A.},
  title     = {Overview of the BioCreative VI chemical–protein interaction Track},
  journal   = {Proceedings of the BioCreative VI Workshop,},
  volume    = {141–146},
  year      = {2017},
  url       = {https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vi/track-5/},
  doi       = {},
  biburl    = {},
  bibsource = {}
}
"""
_DESCRIPTION = """\
The BioCreative VI Chemical-Protein interaction dataset identifies entities of chemicals and proteins and their likely relation to one other. Compounds are generally agonists (activators) or antagonists (inhibitors) of proteins.
"""

_HOMEPAGE = "https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vi/track-5/"

_LICENSE = "Public Domain Mark 1.0"

_URLs = {
    "source": "https://biocreative.bioinformatics.udel.edu/media/store/files/2017/ChemProt_Corpus.zip",
    "bigbio_kb": "https://biocreative.bioinformatics.udel.edu/media/store/files/2017/ChemProt_Corpus.zip",
}

_SUPPORTED_TASKS = [Tasks.RELATION_EXTRACTION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.NAMED_ENTITY_DISAMBIGUATION]
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class ChemprotDataset(datasets.GeneratorBasedBuilder):
    """BioCreative VI Chemical-Protein Interaction Task."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="chemprot_source",
            version=SOURCE_VERSION,
            description="chemprot source schema",
            schema="source",
            subset_id="chemprot",
        ),
        BigBioConfig(
            name="chemprot_bigbio_kb",
            version=BIGBIO_VERSION,
            description="chemprot BigBio schema",
            schema="bigbio_kb",
            subset_id="chemprot",
        ),
    ]

    DEFAULT_CONFIG_NAME = "chemprot_source"

    def _info(self):

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "pmid": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "entities": datasets.Sequence(
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "text": datasets.Value("string"),
                            "offsets": datasets.Sequence(datasets.Value("int64")),
                        }
                    ),
                    "relations": datasets.Sequence(
                        {
                            "type": datasets.Value("string"),
                            "arg1": datasets.Value("string"),
                            "arg2": datasets.Value("string"),
                        }
                    ),
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

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        my_urls = _URLs[self.config.schema]
        data_dir = dl_manager.download_and_extract(my_urls)

        # Extract each of the individual folders
        # NOTE: omitting "extract" call cause it uses a new folder
        train_path = dl_manager.extract(os.path.join(data_dir, "ChemProt_Corpus/chemprot_training.zip"))
        test_path = dl_manager.extract(os.path.join(data_dir, "ChemProt_Corpus/chemprot_test_gs.zip"))
        dev_path = dl_manager.extract(os.path.join(data_dir, "ChemProt_Corpus/chemprot_development.zip"))
        sample_path = dl_manager.extract(os.path.join(data_dir, "ChemProt_Corpus/chemprot_sample.zip"))

        return [
            datasets.SplitGenerator(
                name="sample",  # should be a named split : /
                gen_kwargs={
                    "filepath": os.path.join(sample_path, "chemprot_sample"),
                    "abstract_file": "chemprot_sample_abstracts.tsv",
                    "entity_file": "chemprot_sample_entities.tsv",
                    "relation_file": "chemprot_sample_gold_standard.tsv",
                    "split": "sample",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(train_path, "chemprot_training"),
                    "abstract_file": "chemprot_training_abstracts.tsv",
                    "entity_file": "chemprot_training_entities.tsv",
                    "relation_file": "chemprot_training_gold_standard.tsv",
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(test_path, "chemprot_test_gs"),
                    "abstract_file": "chemprot_test_abstracts_gs.tsv",
                    "entity_file": "chemprot_test_entities_gs.tsv",
                    "relation_file": "chemprot_test_gold_standard.tsv",
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(dev_path, "chemprot_development"),
                    "abstract_file": "chemprot_development_abstracts.tsv",
                    "entity_file": "chemprot_development_entities.tsv",
                    "relation_file": "chemprot_development_gold_standard.tsv",
                    "split": "dev",
                },
            ),
        ]

    def _generate_examples(self, filepath, abstract_file, entity_file, relation_file, split):
        """Yields examples as (key, example) tuples."""
        if self.config.schema == "source":

            abstracts = self._get_abstract(os.path.join(filepath, abstract_file))

            entities, entity_id = self._get_entities(os.path.join(filepath, entity_file))

            relations = self._get_relations(os.path.join(filepath, relation_file), entity_id)

            # NOTE: Not all relations have a gold standard (i.e. annotated by human curators).
            empty_reln = [
                {
                    "type": None,
                    "arg1": None,
                    "arg2": None,
                }
            ]
            for id_, pmid in enumerate(abstracts.keys()):
                yield id_, {
                    "pmid": pmid,
                    "text": abstracts[pmid],
                    "entities": entities[pmid],
                    "relations": relations.get(pmid, empty_reln),
                }

        if self.config.schema == "bigbio_kb":

            abstracts = self._get_abstract(os.path.join(filepath, abstract_file))
            entities, entity_id = self._get_entities(os.path.join(filepath, entity_file))
            relations = self._get_relations(os.path.join(filepath, relation_file), entity_id)

            uid = 0
            for id_, pmid in enumerate(abstracts.keys()):
                data = {
                    "id": str(uid),
                    "document_id": str(pmid),
                    "passages": [],
                    "entities": [],
                    "relations": [],
                    "events": [],
                    "coreferences": [],
                }
                uid += 1

                data["passages"] = [
                    {
                        "id": str(uid),
                        "type": "title and abstract",
                        "text": [abstracts[pmid]],
                        "offsets": [[0, len(abstracts[pmid])]],
                    }
                ]
                uid += 1

                for entity in entities[pmid]:
                    _text = entity["text"]
                    entity.update({"text": [_text]})
                    entity.update({"id": str(uid)})
                    _offsets = entity["offsets"]
                    entity.update({"offsets": [_offsets]})
                    entity.update({"normalized": [{"db_name": "Pubmed", "db_id": str(pmid)}]})
                    data["entities"].append(entity)
                    uid += 1

                empty_reln = [
                    {
                        "type": None,
                        "arg1": None,
                        "arg2": None,
                    }
                ]
                for relation in relations.get(pmid, empty_reln):
                    relation["arg1_id"] = relation.pop("arg1")
                    relation["arg2_id"] = relation.pop("arg2")
                    relation.update({"id": str(uid)})
                    relation.update({"normalized": [{"db_name": "Pubmed", "db_id": str(pmid)}]})
                    data["relations"].append(relation)
                    uid += 1

                yield id_, data

    @staticmethod
    def _get_abstract(abs_filename: str) -> Dict[str, str]:
        """
        For each document in PubMed ID (PMID) in the ChemProt abstract data file, return the abstract. Data is tab-separated.

        :param filename: `*_abstracts.tsv from ChemProt

        :returns Dictionary with PMID keys and abstract text as values.
        """
        with open(abs_filename, "r") as f:
            contents = [i.strip() for i in f.readlines()]

        # PMID is the first column, Abstract is last
        return {doc.split("\t")[0]: "\n".join(doc.split("\t")[1:]) for doc in contents}  # Includes title as line 1

    @staticmethod
    def _get_entities(ents_filename: str) -> Tuple[Dict[str, str]]:
        """
        For each document in the corpus, return entity annotations per PMID.
        Each column in the entity file is as follows:
        (1) PMID
        (2) Entity Number
        (3) Entity Type (Chemical, Gene-Y, Gene-N)
        (4) Start index
        (5) End index
        (6) Actual text of entity

        :param ents_filename: `_*entities.tsv` file from ChemProt

        :returns: Dictionary with PMID keys and entity annotations.
        """
        with open(ents_filename, "r") as f:
            contents = [i.strip() for i in f.readlines()]

        entities = {}
        entity_id = {}

        for line in contents:

            pmid, idx, label, start_offset, end_offset, name = line.split("\t")

            # Populate entity dictionary
            if pmid not in entities:
                entities[pmid] = []

            ann = {
                "offsets": [int(start_offset), int(end_offset)],
                "text": name,
                "type": label,
                "id": idx,
            }

            entities[pmid].append(ann)

            # Populate entity mapping
            entity_id.update({idx: name})

        return entities, entity_id

    @staticmethod
    def _get_relations(rel_filename: str, ent_dict: Dict[str, str]) -> Dict[str, str]:
        """
        For each document in the ChemProt corpus, create an annotation for the gold-standard relationships.

        The columns include:
        (1) PMID
        (2) Relationship Label (CPR)
        (3) Interactor Argument 1 Entity Identifier
        (4) Interactor Argument 2 Entity Identifier

        Gold standard includes CPRs 3-9. Relationships are always Gene + Protein.
        Unlike entities, there is no counter, hence once must be made

        :param rel_filename: Gold standard file name
        :param ent_dict: Entity Identifier to text
        """
        with open(rel_filename, "r") as f:
            contents = [i.strip() for i in f.readlines()]

        relations = {}

        for line in contents:
            pmid, label, arg1, arg2 = line.split("\t")
            arg1 = arg1.split("Arg1:")[-1]
            arg2 = arg2.split("Arg2:")[-1]

            if pmid not in relations:
                relations[pmid] = []

            ann = {
                "type": label,
                "arg1": ent_dict.get(arg1, None),
                "arg2": ent_dict.get(arg2, None),
            }

            relations[pmid].append(ann)

        return relations


if __name__ == "__main__":
    from datasets import load_dataset

    # ds = load_dataset(__file__)
    ds = load_dataset(__file__)
    print(ds)
