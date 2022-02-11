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
BioASQ Task B On Biomedical Semantic QA (Involves IR, QA, Summarization qnd
More). This task uses benchmark datasets containing development and test
questions, in English, alongwith gold standard (reference) answers constructed
by a team of biomedical experts. The participants have to respond with relevant
concepts, articles, snippets and RDF triples, fromdesignated resources, as well
as exact and 'ideal' answers.

Fore more information about the challenge, the organisers and the relevant
publications please visit: http://bioasq.org/
"""
import json
import os
import glob
import datasets


_CITATION = """\
@article{tsatsaronis2015overview,
	title        = {
		An overview of the BIOASQ large-scale biomedical semantic indexing and
		question answering competition
	},
	author       = {
		Tsatsaronis, George and Balikas, Georgios and Malakasiotis, Prodromos and
		Partalas, Ioannis and Zschunke, Matthias and Alvers, Michael R and
		Weissenborn, Dirk and Krithara, Anastasia and Petridis, Sergios and
		Polychronopoulos, Dimitris and others
	},
	year         = 2015,
	journal      = {BMC bioinformatics},
	publisher    = {BioMed Central Ltd},
	volume       = 16,
	number       = 1,
	pages        = 138
}
"""

_DESCRIPTION = """\
The data are intended to be used as training and development data for BioASQ 9, which will
take place during 2021. There is one file containing the data:
 - training9b.json

The file contains the data of the first seven editions of the challenge: 3742 questions [1]
with their relevant documents, snippets, concepts and RDF triples, exact and ideal answers.
For more information about the format of the data as well as the instructions for
participating at BioASQ please consult: 
http://participants-area.bioasq.org/general_information/Task9b/

Differences with BioASQ-training8b.json 
	- 499 new questions added from BioASQ8
		- The question with id 5e30e689fbd6abf43b00003a had identical body with 
          5880e417713cbdfd3d000001. All relevant elements from both questions are available
          in the merged question with id 5880e417713cbdfd3d000001.
"""

_HOMEPAGE = "http://participants-area.bioasq.org/datasets/"

# Data access reqires registering with BioASQ.
# See http://participants-area.bioasq.org/accounts/register/
_LICENSE = ""

_URLs = {"train": "BioASQ-training9b.zip", "test": "Task9BGoldenEnriched.zip"}


class Bioasq9bDataset(datasets.GeneratorBasedBuilder):
    """BioASQ Task B On Biomedical Semantic QA."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="original",
            version=VERSION,
            description="Original BioASQ9 source schema.",
        ),
        datasets.BuilderConfig(
            name="bigbio",
            version=VERSION,
            description="Simplified schema for BigScience-Biomedical QA tasks",
        ),
    ]

    DEFAULT_CONFIG_NAME = "original"

    def _info(self):

        # BioASQ9 Task B original schema
        if self.config.name == "original":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "type": datasets.Value("string"),
                    "body": datasets.Value("string"),
                    "documents": datasets.Sequence(datasets.Value("string")),
                    "concepts": datasets.Sequence(datasets.Value("string")),
                    "ideal_answer": datasets.Sequence(datasets.Value("string")),
                    "exact_answer": datasets.Sequence(datasets.Value("string")),
                    "triples": datasets.Sequence(
                        {
                            "p": datasets.Value("string"),
                            "s": datasets.Value("string"),
                            "o": datasets.Value("string"),
                        }
                    ),
                    "snippets": datasets.Sequence(
                        {
                            "offsetInBeginSection": datasets.Value("int32"),
                            "offsetInEndSection": datasets.Value("int32"),
                            "text": datasets.Value("string"),
                            "beginSection": datasets.Value("string"),
                            "endSection": datasets.Value("string"),
                            "document": datasets.Value("string"),
                        }
                    ),
                }
            )
        # simplified scheam for QA tasks
        elif self.config.name == "bigbio":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "question_id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "type": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "answer": datasets.Sequence(datasets.Value("string")),
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
        data_dir = dl_manager.download_and_extract(list(_URLs.values()))
        # BioASQ test data is split into multiple records {9B1_golden.json,...,9B5_golden.json}
        # combine these files into a single test set 9Bx_golden.json
        if not os.path.exists(
            os.path.join(data_dir, "Task9BGoldenEnriched/9Bx_golden.json")
        ):
            pass
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, "BioASQ-training9b/training9b.json"
                    ),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, "BioASQ-training9b/training9b.json"
                    ),
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(
        self,
        filepath,
        split,  # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    ):
        """Yields examples as (key, example) tuples."""
        # This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is here for legacy reason (tfds) and is not important in itself.

        with open(filepath, encoding="utf-8") as file:
            data = json.load(file)
            for record in enumerate(data):
                if self.config.name == "original":
                    yield record["id"], {
                        "id": record["id"],
                        "type": record["type"],
                        "body": record["body"],
                        "documents": record["documents"],
                        "concepts": record["concepts"] if "concepts" in record else [],
                        "triples": record["triplets"] if "triples" in record else [],
                        "ideal_answer": record["ideal_answer"],
                        "exact_answer": record["exact_answer"],
                        "snippets": record["snippets"],
                    }

                elif self.config.name == "bigbio":
                    # instances are defined over snippets/documents
                    for i, snippet in enumerate(record["snippets"]):
                        uid = f'{record["id"]}_{i}'
                        yield uid, {
                            "id": record["id"],
                            "document_id": snippet["document"],
                            "question_id": record["id"],
                            "question": record["body"],
                            "type": record["type"],
                            "context": snippet["text"],
                            "answer": record["exact_answer"],
                        }
