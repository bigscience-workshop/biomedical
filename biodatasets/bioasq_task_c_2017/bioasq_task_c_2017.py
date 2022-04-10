# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and
#
# * <append your name and optionally your github handle here>
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

"""


import json
import os
from pathlib import Path
import sys
import time

sys.path.append("/Users/patrickhaller/Projects/biomedical/")
from typing import List

import datasets
from tqdm import tqdm
from Bio import Entrez

from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

_CITATION = """\
@article{nentidis-etal-2017-results,
  author    = {Nentidis, Anastasios  and Bougiatiotis, Konstantinos  and Krithara, Anastasia  and
               Paliouras, Georgios  and Kakadiaris, Ioannis},
  title     = {Results of the fifth edition of the {B}io{ASQ} Challenge},
  journal   = {},
  volume    = {BioNLP 2017},
  year      = {2007},
  url       = {https://aclanthology.org/W17-2306},
  doi       = {10.18653/v1/W17-2306},
  biburl    = {},
  bibsource = {https://aclanthology.org/W17-2306}
}
"""

# TODO: create a module level variable with your dataset name (should match script name)
_DATASETNAME = "bioasq_task_c_2017"

_DESCRIPTION = """\
The training data set for this task contains annotated biomedical articles published in PubMed and corresponding full text from PMC. By annotated is meant that GrantIDs and corresponding Grant Agencies have been identified in the full text of articles.
"""

_HOMEPAGE = "http://participants-area.bioasq.org/general_information/Task5c/"

_LICENSE = "NLM License Code: 8283NLM123"

# Website contains all data, but login required
_URLS = {
    _DATASETNAME: "http://participants-area.bioasq.org/datasets/"
}

_SUPPORTED_TASKS = [Tasks.TEXT_CLASSIFICATION]

# this version doesn't have to be consistent with semantic versioning. Anything that is
# provided by the original dataset as a version goes.
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class BioASQTaskC2017(datasets.GeneratorBasedBuilder):
    """
    BioASQ Task C on 
    """

    DEFAULT_CONFIG_NAME = "bioasq_task_c_2017_source"
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="bioasq_task_c_2017_source",
            version=SOURCE_VERSION,
            description="bioasq_task_c_2017 source schema",
            schema="source",
            subset_id="bioasq_task_c_2017",
            # cred_mail=""
            
        ),
        BigBioConfig(
            name="bioasq_task_c_2017_bigbio_text",
            version=BIGBIO_VERSION,
            description="bioasq_task_c_2017 BigBio schema",
            schema="bigbio_text",
            subset_id="bioasq_task_c_2017",
            # cred_mail=""
        )
    ]


    def _info(self) -> datasets.DatasetInfo:

        # BioASQ Task C source schema
        if self.config.schema == "source":
            features = datasets.Features(
               {
                   "document_id": datasets.Value("string"),
                   "pmid": datasets.Value("string"),
                   "pmcid": datasets.Value("string"),
                   "grantList": [
                       {
                           # "grantID": datasets.Value("string"),
                           "agency": datasets.Value("string"),
                       }
                   ],
                   "text": datasets.Value("string")
               }
            )

        # For example bigbio_kb, bigbio_t2t
        elif self.config.schema =="bigbio_text":
            features = schemas.text

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration

        # If you need to access the "source" or "bigbio" config choice, that will be in self.config.name

        # LOCAL DATASETS: You do not need the dl_manager; you can ignore this argument. Make sure `gen_kwargs` in the return gets passed the right filepath

        # PUBLIC DATASETS: Assign your data-dir based on the dl_manager.

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLs; many examples use the download_and_extract method; see the DownloadManager docs here: https://huggingface.co/docs/datasets/package_reference/builder_classes.html#datasets.DownloadManager

        # dl_manager can accept any type of nested list/dict and will give back the same structure with the url replaced with the path to local files.

        # # TODO: KEEP if your dataset is PUBLIC; REMOVE if not
        # urls = _URLS[_DATASETNAME]
        # data_dir = dl_manager.download_and_extract(urls)

        if self.config.data_dir is None:
            raise ValueError("This is a local dataset. Please pass the data_dir kwarg to load_dataset.")
        else:
            data_dir = self.config.data_dir

        # Not all datasets have predefined canonical train/val/test splits. If your dataset does not have any splits, you can omit any missing splits.
        # If your dataset has no predefined splits, use datasets.Split.TRAIN for all of the data.

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "taskCTrainingData2017.json"),
                    "filespath": os.path.join(data_dir, "Train_Text"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "taskc_golden2.json"),
                    "filespath": os.path.join(data_dir, "Final_Text"),
                    "split": "test",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "taskc_golden2.json"),
                    "filespath": os.path.join(data_dir, "Final_Text"),
                    "split": "dev",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`

    def _generate_examples(self, filepath, filespath, split) -> (int, dict):

        with open(filepath) as f:
            task_data = json.load(f)

        # pmids = []
        # for article in task_data["articles"]:
        #     pmids.append(article["pmid"])
        #
        # Download all pubmed articles
        # pubmed_dir = self.config.data_dir + "/.pubmed_data"
        # fetch_pubmed_abstracts(pmids, pubmed_dir, self.config.cred_mail)

        # articles = {}
        #
        # for pubmed_file in Path(pubmed_dir).iterdir():
        #
        #     with open(pubmed_file) as f:
        #         pubmed_json = json.load(f)
        #     
        #     # Collect article according to pmid
        #     for article in pubmed_json["PubmedArticle"]:
        #         pmid = article["MedlineCitation"]["PMID"]
        #         articles[pmid] = article

        
        if self.config.schema == "source":
            for article in task_data["articles"]:
                with open(filespath + "/" + article["pmcid"] + ".xml") as f:
                    text = f.read()
                    
                pmid = article["pmid"]

                yield pmid, {
                    "text": text, #articles[pmid],
                    "document_id": pmid,
                    "pmid": pmid,
                    "pmcid": article["pmcid"],
                    "grantList": [{"agency": grant["agency"]} for grant in article["grantList"]]
                }

        elif self.config.schema == "bigbio":
            for article in task_data["articles"]:
                with open(filespath + "/" + article["pmcid"] + ".xml") as f:
                    text = f.read()
                
                yield article["pmid"], {
                    "text": text,
                    "id": article["pmid"],
                    "document_id": article["pmid"],
                    "labels": [grant["agency"] for grant in article["grantList"]]
                }

def fetch_pubmed_abstracts(
    pmids: List,
    outdir: str,
    cred_mail: str,
    batch_size: int = 1000,
    delay: float = 0.3,
    overwrite: bool = False,
    verbose: bool = True,
):

    Entrez.email = cred_mail

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    if not isinstance(pmids, list):
        pmids = list(pmids)

    n_chunks = int(len(pmids) / batch_size)

    for i in tqdm(
        range(0, n_chunks + (1 if n_chunks * batch_size < len(pmids) else 0))
    ):
        start, end = i * batch_size, (i + 1) * batch_size
        outfname = f"{outdir}/pubmed.{i}.json"
        if os.path.exists(outfname) and not overwrite:
            continue

        query = ",".join(pmids[start:end])
        handle = Entrez.efetch(
            db="pubmed", id=query, rettype="gb", retmode="xml", retmax=batch_size
        )
        record = Entrez.read(handle)
        if len(record["PubmedArticle"]) != len(pmids[start:end]) and verbose:
            print(
                f"Queried {len(pmids[start:end])}, returned {len(record['PubmedArticle'])}"
            )


        time.sleep(delay)
        # dump to JSON
        with open(outfname, "wt") as file:
            file.write(json.dumps(record, indent=2))

# This allows you to run your dataloader with `python [dataset_name].py` during development
# TODO: Remove this before making your PR
if __name__ == "__main__":
    datasets.load_dataset(__file__, name="bioasq_task_c_2017_source", data_dir="/Users/patrickhaller/Projects/biomedical/data")
