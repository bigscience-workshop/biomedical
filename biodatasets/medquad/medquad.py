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
import glob
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict

import datasets
from pyarrow.dataset import dataset

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

_TEST_DATA_PATH = "https://raw.githubusercontent.com/abachaa/LiveQA_MedicalTask_TREC2017/master/TestDataset/TREC-2017-LiveQA-Medical-Test.xml"

_DATA_REPO_FETCH_URL = "https://github.com/abachaa/MedQuAD/archive/refs/heads/master.zip"

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
    "medquad_base_uris": _SUBSET_BASE_URIS,
    "medquad_test_base_uris": _TEST_DATA_PATH,
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
                    "Answer": datasets.Sequence(datasets.Value("string")),
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

    def _load_train_qa_from_xml(self, file_paths) -> List[dict[str, str | None]]:
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

        return qa_list

    def _load_test_qa_from_xml(self, file_paths) -> List[dict[str, str | None]]:
        """
        This method traverses the downloaded test XML file and extracts Q&A pairs.
        Returns the extracted Q&As and the base directory of the dumped json file that contains them all.
        """
        assert len(file_paths)

        qa_list = []
        for file_path in file_paths:
            doc_root = ET.parse(file_path).getroot()
            for nlm_quetion in doc_root:
                qid = nlm_quetion.attrib.get("qid")

                original_question = nlm_quetion[0]
                original_question_qfile = original_question.attrib.get("qfile")
                original_question_subject = original_question[0]
                original_question_message = original_question[1]
                nist_paraphrase = nlm_quetion[1]
                annotations = nlm_quetion[2]
                annotation_focuses, annotation_types, annotation_keywords = [], [], []
                for annotation in annotations:
                    if annotation.tag == "FOCUS":
                        annotation_focuses.append({
                            "fid": annotation.attrib.get("fid"),
                            "fcategory": annotation.attrib.get("fcategory"),
                            "text": annotation.text,
                        })
                    elif annotation.tag == "TYPE":
                        annotation_focuses.append({
                            "tid": annotation.attrib.get("tid"),
                            "hasFocus": annotation.attrib.get("hasFocus"),
                            "hasKeyword": annotation.attrib.get("hasKeyword"),
                            "text": annotation.text,
                        })
                    elif annotation.tag == "KEYWORD":
                        annotation_focuses.append({
                            "kid": annotation.attrib.get("kid"),
                            "kcategory": annotation.attrib.get("kcategory"),
                            "text": annotation.text,
                        })

                reference_answers = nlm_quetion[3]
                ref_answers = []
                for ref_answer in reference_answers:
                    ref_answers.append({
                        "aid": ref_answer.attrib.get("aid"),
                        "ANSWER": ref_answer[0].text,
                        "AnswerURL": ref_answer[1].text,
                        "COMMENT": ref_answer[2].text,
                    })

                qa_list.append({
                    "qid": qid,
                    "Original-Question": {
                        "qfile": original_question_qfile,
                        "SUBJECT": original_question_subject.text,
                        "MESSAGE": original_question_message.text,
                    },
                    "NIST-PARAPHRASE": nist_paraphrase.text,
                    "ANNOTATIONS": {
                        "FOCUS": annotation_focuses,
                        "TYPE": annotation_types,
                        "KEYWORD": annotation_keywords,
                    },
                    "ReferenceAnswers": ref_answers,
                })

        return qa_list

    def _dump_xml_to_json(self, qa_file_paths, split):

        qa_pairs_enriched_fname = f"MedQuADGoldenEnriched/{self.config.subset_id}_{split}.json"

        # Collect path info for all repo paths, and determine relevant XML files
        data_dir = os.path.dirname(qa_file_paths[0])

        qa_pairs_enriched_full_path = os.path.join(data_dir, qa_pairs_enriched_fname)

        if not os.path.exists(qa_pairs_enriched_full_path):
            if split == datasets.Split.TEST:
                qa_list = self._load_test_qa_from_xml(
                    file_paths=qa_file_paths
                )
            else:
                qa_list = self._load_train_qa_from_xml(
                    file_paths=qa_file_paths
                )

            qa_pairs_enriched_dir = os.path.dirname(qa_pairs_enriched_full_path)
            if not os.path.exists(qa_pairs_enriched_dir):
                os.mkdir(qa_pairs_enriched_dir)

            # dump QA paris to json
            with open(qa_pairs_enriched_full_path, "wt", encoding="utf-8") as file:
                json.dump(qa_list, file)

        return qa_pairs_enriched_full_path

    def _dump_test_xml_to_json(self, dl_manager):
        file_base_url = _URLS["medquad_test_base_uris"]
        file_extracted = dl_manager.download_and_extract(file_base_url)

        return self._dump_xml_to_json([file_extracted], split=datasets.Split.TEST)

    def _dump_train_xml_to_json(self, dl_manager) -> str:
        """
        This method parses training dataset, or a single batch that belongs to the websites,
        please check the repo page.
        """
        repo_extracted = dl_manager.download_and_extract(_DATA_REPO_FETCH_URL)
        repo_dir = os.path.join(
            repo_extracted,
            os.path.basename(_HOMEPAGE) + '-' + os.path.splitext(os.path.basename(_DATA_REPO_FETCH_URL))[0]
        )

        if self.config.subset_id == "medquad":
            file_base_urls = _URLS[f"medquad_base_uris"]
        else:
            file_base_urls = [_SUBSET_BASE_URIS[self.config.subset_id]]

        qa_file_paths = []
        for subset_name, uri_ in file_base_urls.items():
            for file_path in glob.glob(os.path.join(repo_dir, uri_, "*.xml")):
                qa_file_paths.append(file_path)

        return self._dump_xml_to_json(qa_file_paths, split=datasets.Split.TRAIN)

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        qa_pairs_enriched_train_fpath = self._dump_train_xml_to_json(dl_manager)
        qa_pairs_enriched_test_fpath = self._dump_test_xml_to_json(dl_manager)

        # There is no canonical train/valid/test set in this dataset. So, only TRAIN is added.
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": qa_pairs_enriched_train_fpath,
                    "split": datasets.Split.TRAIN,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": qa_pairs_enriched_test_fpath,
                    "split": datasets.Split.TEST,
                },
            ),
        ]

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        with open(filepath, encoding="utf-8") as file:
            data = json.load(file)
            if self.config.schema == "source":
                for key, record in enumerate(data):
                    if split == datasets.Split.TEST:
                        yield key, {
                            "Document": None,
                            "QAPair": None,
                            "qid": record["qid"],
                            "qtype": record["Original-Question"]["SUBJECT"],
                            # A paraphrased verions of record["Original-Question"]["MESSAGE"]
                            "Question": record["NIST-PARAPHRASE"],
                            "Answer": [ref["ANSWER"] for ref in record["ReferenceAnswers"]],
                        }
                    else:
                        yield key, {
                            "Document": record["Document"],
                            "QAPair": record["QAPair"],
                            "qid": record["qid"],
                            "qtype": record["qtype"],
                            "Question": record["Question"],
                            "Answer": [record["Answer"]],
                        }

            elif self.config.schema == "bigbio_qa":
                uid = 0
                for key, record in enumerate(data):
                    uid += 1
                    yield key, {
                        "id": str(uid),
                        "document_id": record.get("Document"),
                        "question_id": record["qid"],
                        "question": record.get("Question"),
                        "type": record.get("qtype"),
                        "choices": [],
                        "context": [],
                        "answer": [record.get("Answer")],
                    }
