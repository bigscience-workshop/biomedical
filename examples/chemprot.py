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

import os
from typing import Dict, Tuple

import datasets


_DATASETNAME = "chemprot"

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

_URLs = {"chemprot": "https://biocreative.bioinformatics.udel.edu/media/store/files/2017/ChemProt_Corpus.zip"}

_VERSION = "1.0.0"


class ChemprotDataset(datasets.GeneratorBasedBuilder):
    """BioCreative VI Chemical-Protein Interaction Task."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=_DATASETNAME,
            version=VERSION,
            description=_DESCRIPTION,
        ),
    ]

    DEFAULT_CONFIG_NAME = (
        _DATASETNAME  # It's not mandatory to have a default configuration. Just use one if it make sense.
    )

    def _info(self):

        if self.config.name == _DATASETNAME:
            features = datasets.Features(
                {
                    "pmid": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "entities": datasets.Sequence(
                        {
                            "offsets": datasets.Sequence(datasets.Value("int64")),
                            "text": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "entity_id": datasets.Value("string"),
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

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLs
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        my_urls = _URLs[self.config.name]
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
        if self.config.name == _DATASETNAME:

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
                "entity_id": idx,
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
