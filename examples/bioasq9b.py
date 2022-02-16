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

_URLs = ["BioASQ-training9b.zip", "Task9BGoldenEnriched.zip"]


class Bioasq9bDataset(datasets.GeneratorBasedBuilder):
    """BioASQ Task B On Biomedical Semantic QA."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="source",
            version=VERSION,
            description="Original BioASQ9 source schema.",
        ),
        datasets.BuilderConfig(
            name="bigbio",
            version=VERSION,
            description="Simplified schema for BigScience-Biomedical QA tasks",
        ),
    ]

    DEFAULT_CONFIG_NAME = "source"

    def _info(self):

        # BioASQ9 Task B source schema
        if self.config.name == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "type": datasets.Value("string"),
                    "body": datasets.Value("string"),
                    "documents": datasets.Sequence(datasets.Value("string")),
                    "concepts": datasets.Sequence(datasets.Value("string")),
                    "ideal_answer": datasets.Sequence(datasets.Value("string")),
                    "exact_answer": datasets.Sequence(datasets.Value("string")),
                    "triples": [
                        {
                            "p": datasets.Value("string"),
                            "s": datasets.Value("string"),
                            "o": datasets.Value("string"),
                        }
                    ],
                    "snippets": [
                        {
                            "offsetInBeginSection": datasets.Value("int32"),
                            "offsetInEndSection": datasets.Value("int32"),
                            "text": datasets.Value("string"),
                            "beginSection": datasets.Value("string"),
                            "endSection": datasets.Value("string"),
                            "document": datasets.Value("string"),
                        }
                    ],
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
        train_dir, test_dir = dl_manager.download_and_extract(_URLs)

        # BioASQ test data is split into multiple records {9B1_golden.json,...,9B5_golden.json}
        # combine these files into a single test set file 9Bx_golden.json
        test_fpath = os.path.join(test_dir, "Task9BGoldenEnriched/9Bx_golden.json")

        if not os.path.exists(test_fpath):
            filelist = glob.glob(os.path.join(test_dir, "Task9BGoldenEnriched/*.json"))
            test_data = None
            for fname in sorted(filelist):
                data = json.load(open(fname, "rt"))
                if test_data is None:
                    test_data = data
                else:
                    test_data["questions"].extend(data["questions"])
            json.dump(data, open(test_fpath, "wt"))

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(
                        train_dir, "BioASQ-training9b/training9b.json"
                    ),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": os.path.join(
                        test_dir, "Task9BGoldenEnriched/9Bx_golden.json"
                    ),
                    "split": "test",
                },
            ),
        ]

    def _get_exact_answer(self, record):
        """The value exact_answer can be in different formats based on question type."""
        if record["type"] == "yesno":
            exact_answer = [record["exact_answer"]]
        elif record["type"] == "summary":
            exact_answer = []
        elif record["type"] == "list":
            exact_answer = record["exact_answer"]
        elif record["type"] == "factoid":
            exact_answer = record["exact_answer"]
        return exact_answer

    def _generate_examples(
        self,
        filepath,
        split,  # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    ):
        """Yields examples as (key, example) tuples."""
        if self.config.name == "source":
            with open(filepath, encoding="utf-8") as file:
                data = json.load(file)
                for key, record in enumerate(data["questions"]):
                    yield key, {
                        "id": record["id"],
                        "type": record["type"],
                        "body": record["body"],
                        "documents": record["documents"],
                        "concepts": record["concepts"] if "concepts" in record else [],
                        "triples": record["triples"] if "triples" in record else [],
                        "ideal_answer": record["ideal_answer"],
                        "exact_answer": self._get_exact_answer(record),
                        "snippets": record["snippets"],
                    }

        elif self.config.name == "bigbio":
            with open(filepath, encoding="utf-8") as file:
                data = json.load(file)
                for key, record in enumerate(data["questions"]):
                    for key, snippet in enumerate(record["snippets"]):
                        uid = f'{record["id"]}_{key}'
                        yield uid, {
                            "id": uid,
                            "document_id": snippet["document"],
                            "question_id": record["id"],
                            "question": record["body"],
                            "type": record["type"],
                            "context": snippet["text"],
                            # summary question types only have an idea answer
                            "answer": self._get_exact_answer(record),
                        }
