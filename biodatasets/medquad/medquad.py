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
This is an implementation of the dataloader for MedQuAD dataset.
MedQuAD includes 47,457 medical question-answer pairs created from 12 NIH websites (e.g. cancer.gov, niddk.nih.gov, GARD, MedlinePlus Health Topics). The collection covers 37 question types (e.g. Treatment, Diagnosis, Side Effects) associated with diseases, drugs and other medical entities such as tests.
We included additional annotations in the XML files, that could be used for diverse IR and NLP tasks, such as the question type, the question focus, its syonyms, its UMLS Concept Unique Identifier (CUI) and Semantic Type.
We  added the category of the question focus (Disease, Drug or Other) in the 4 MedlinePlus collections. All other collections are about diseases.
The paper cited below describes the collection, the construction method as well as its use and evaluation within a medical question answering system.
N.B. We removed the answers from 3 subsets to respect the MedlinePlus copyright (https://medlineplus.gov/copyright.html):
(1) A.D.A.M. Medical Encyclopedia, (2) MedlinePlus Drug information, and (3) MedlinePlus Herbal medicine and supplement information.
-- We kept all the other information including the URLs in case you want to crawl the answers. Please contact me if you have any questions.

For more info please visit https://github.com/abachaa/MedQuAD/
"""
import json
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict

import datasets
import requests

from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

_CITATION = """\
@article{BenAbacha-BMC-2019,
  author    = {Asma {Ben Abacha} and Dina Demner{-}Fushman},
  title     = {A Question-Entailment Approach to Question Answering},
  journal   = {{BMC} Bioinform.},
  volume    = {20},
  pages     = {511:1--511:23},
  year      = {2019},
  url       = {https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-3119-4},
  doi       = {https://doi.org/10.1186/s12859-019-3119-4},
  biburl    = {https://citation-needed.springer.com/v2/references/10.1186/s12859-019-3119-4?format=refman&flavour=citation},
}
"""

_DATASETNAME = "medquad"

_DESCRIPTION = """\
MedQuAD: Medical Question Answering Dataset  
MedQuAD includes 47,457 medical question-answer pairs created from 12 NIH websites\
(e.g. cancer.gov, niddk.nih.gov, GARD, MedlinePlus Health Topics).\
The collection covers 37 question types (e.g. Treatment, Diagnosis, Side Effects) associated with diseases,\
drugs and other medical entities such as tests.
"""

_HOMEPAGE = "https://github.com/abachaa/MedQuAD"

_LICENSE = "https://creativecommons.org/licenses/by/4.0/legalcode"  # TODO: terms aren't available in the repository! In the issue it is 'CC BY 4.0'

_DATA_PATH = "https://raw.githubusercontent.com/abachaa/MedQuAD/master"

_DATA_REPO_FETCH_URL = "https://api.github.com/repos/abachaa/MedQuAD/git/trees/master?recursive=1"

_SUBSET_BASE_URIS = {
    "cancergov_qa": "1_CancerGov_QA",
    "gard_qa": "2_GARD_QA",
    "ghr_qa": "3_GHR_QA",
    "mplus_health_topics_qa": "4_MPlus_Health_Topics_QA",
    "niddk_qa": "5_NIDDK_QA",
    "ninds_qa": "6_NINDS_QA",
    "seniorhealth_qa": "7_SeniorHealth_QA",
    "nhlbi_qa_xml": "8_NHLBI_QA_XML",
    "cdc_qa": "9_CDC_QA",
    "mplus_adam_qa": "10_MPlus_ADAM_QA",
    "mplusdrugs_qa": "11_MPlusDrugs_QA",
    "mplusherbssupplements_qa": "12_MPlusHerbsSupplements_QA",
}

_URLS = {
    "medquad_base_urls": _SUBSET_BASE_URIS,
    f"QATestSetMedQrels_judged_answers": f"{_DATA_PATH}/QA-TestSet-LiveQA-Med-Qrels-2479-Answers.zip",
}

_SUPPORTED_TASKS = [Tasks.QUESTION_ANSWERING]  # TODO: shall we add a non-existing task type such as `RQE`?

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class MedquadDataset(datasets.GeneratorBasedBuilder):
    """MedQuAD: Medical Question Answering Dataset"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="medquad_source",
            version=SOURCE_VERSION,
            description="medquad source schema",
            schema="source",
            subset_id="medquad",
        ),
        BigBioConfig(
            name="medquad_bigbio_qa",
            version=BIGBIO_VERSION,
            description="medquad BigBio schema",
            schema="bigbio_qa",
            subset_id="medquad",
        ),
    ]

    DEFAULT_CONFIG_NAME = "medquad_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "Document": datasets.Value("string"),
                    "QAPair": datasets.Value("string"),
                    "qid": datasets.Value("string"),
                    "qtype": datasets.Value("string"),
                    "Question": datasets.Value("string"),
                    "Answer": datasets.Value("string"),
                }
            )

        elif self.config.schema == "bigbio_qa":
            features = schemas.qa_features
        else:
            raise NotImplementedError("Only `source` and `bigbio_qa` schemas are implemented.")

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _load_qa_from_xml(self, file_paths) -> Tuple[List[dict], str]:
        """
        This method traverses the whole list of the downloaded XML files and extracts Q&A pairs.
        Returns the extracted Q&As and the base directory of the dumped json file that contains them all.
        """
        assert len(file_paths)

        qa_list = []
        for file_path in file_paths:
            doc_root = ET.parse(file_path).getroot()
            document_id = doc_root.attrib.get("id")
            for element in doc_root:
                if element.tag == "QAPairs":
                    qa_pairs = element
                    for qa_pair in qa_pairs:
                        # Handled this way in case question & answer order occur differently
                        question = qa_pair[0] if qa_pair[0].tag == "Question" else qa_pair[1]
                        answer = qa_pair[1] if qa_pair[1].tag == "Answer" else qa_pair[0]

                        qa_list.append({
                            "Document": document_id,
                            "QAPair": qa_pair.attrib.get("pid"),
                            "qid": question.attrib.get("qid"),
                            "qtype": question.attrib.get("qtype"),
                            "Question": question.text,
                            "Answer": answer.text,
                        })

        return qa_list, os.path.dirname(file_paths[0])

    def _dump_xml_to_json(self, dl_manager) -> str:
        """
        This method parses the dataset

        """
        if self.config.subset_id == "medquad":
            file_base_urls = _URLS[f"{self.config.subset_id}_base_urls"]
            qa_pairs_enriched_fname = f"MedQuADGoldenEnriched/{self.config.subset_id}.json"
            repo_files = json.loads(requests.get(_DATA_REPO_FETCH_URL).text)
        else:
            raise NotImplementedError("Only full set loader is implemented here")

        # Collect path info for all repo paths, and determine relevant XML files
        qa_file_paths = []
        for subset_name, uri_ in file_base_urls.items():
            for path in repo_files["tree"]:
                if path["type"] == "blob" and path["path"].startswith(uri_) and path["path"].endswith(".xml"):
                    qa_file_paths.append(os.path.join(_DATA_PATH, path["path"]))

        qa_list, data_dir = self._load_qa_from_xml(
            file_paths=dl_manager.download_and_extract(qa_file_paths)
        )

        qa_pairs_enriched_full_path = os.path.join(data_dir, qa_pairs_enriched_fname)

        qa_pairs_enriched_dir = os.path.dirname(qa_pairs_enriched_full_path)
        if not os.path.exists(qa_pairs_enriched_dir):
            os.mkdir(qa_pairs_enriched_dir)

        if not os.path.exists(qa_pairs_enriched_full_path):
            data = {"qa_pairs": qa_list}
            # dump QA paris to json
            with open(qa_pairs_enriched_full_path, "wt", encoding="utf-8") as file:
                json.dump(data, file, indent=2)

        return qa_pairs_enriched_full_path

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        qa_pairs_enriched_fpath = self._dump_xml_to_json(dl_manager)

        # There is no canonical train/valid/test set in this dataset. So, only TRAIN is added.
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": qa_pairs_enriched_fpath,
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        if self.config.schema == "source":
            with open(filepath, encoding="utf-8") as file:
                data = json.load(file)
                for key, record in enumerate(data["qa_pairs"]):
                    yield key, {
                        "Document": record["Document"],
                        "QAPair": record["QAPair"],
                        "qid": record["qid"],
                        "qtype": record["qtype"],
                        "Question": record["Question"],
                        "Answer": record["Answer"],
                    }

        elif self.config.schema == "bigbio_qa":
            with open(filepath, encoding="utf-8") as file:
                uid = 0
                data = json.load(file)
                for key, record in enumerate(data["qa_pairs"]):
                    uid += 1
                    yield key, {
                        "id": str(uid),
                        "document_id": record["Document"],
                        "question_id": record["qid"],
                        "question": record["Question"],
                        "type": record["qtype"],
                        "choices": [],
                        "context": [],
                        "answer": [record["Answer"]],
                    }
