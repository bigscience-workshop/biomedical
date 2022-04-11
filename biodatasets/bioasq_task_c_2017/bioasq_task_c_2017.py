# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and
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


import json
import os
import time
from typing import List

import datasets
from tqdm import tqdm

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

_DATASETNAME = "bioasq_task_c_2017"

_DESCRIPTION = """\
The training data set for this task contains annotated biomedical articles published in PubMed
and corresponding full text from PMC. By annotated is meant that GrantIDs and corresponding
Grant Agencies have been identified in the full text of articles.
"""

_HOMEPAGE = "http://participants-area.bioasq.org/general_information/Task5c/"

_LICENSE = "NLM License Code: 8283NLM123"

# Website contains all data, but login required
_URLS = {_DATASETNAME: "http://participants-area.bioasq.org/datasets/"}

_SUPPORTED_TASKS = [Tasks.TEXT_CLASSIFICATION]

# this version doesn't have to be consistent with semantic versioning. Anything that is
# provided by the original dataset as a version goes.
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class BioASQTaskC2017(datasets.GeneratorBasedBuilder):
    """
    BioASQ Task C Dataset for 2017
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
        ),
    ]

    def _info(self) -> datasets.DatasetInfo:

        # BioASQ Task C source schema
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "pmid": datasets.Value("string"),
                    "pmcid": datasets.Value("string"),
                    "grantList": [
                        {
                            "agency": datasets.Value("string"),
                        }
                    ],
                    "text": datasets.Value("string"),
                }
            )

        # For example bigbio_kb, bigbio_t2t
        elif self.config.schema == "bigbio_text":
            features = schemas.text.features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:

        if self.config.data_dir is None:
            raise ValueError("This is a local dataset. Please pass the data_dir kwarg to load_dataset.")
        else:
            data_dir = self.config.data_dir

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
        ]

    def _generate_examples(self, filepath, filespath, split) -> (int, dict):

        with open(filepath) as f:
            task_data = json.load(f)

        # Possible usage of `fetch_pubmed_abstracts` function
        # pmids = []
        # for article in task_data["articles"]:
        #     pmids.append(article["pmid"])
        #
        # Download all pubmed articles
        # pubmed_dir = self.config.data_dir + "/.pubmed_data"
        # fetch_pubmed_abstracts(pmids, pubmed_dir, self.config.cred_mail)
        #
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
            for i, article in enumerate(task_data["articles"]):
                with open(filespath + "/" + article["pmcid"] + ".xml") as f:
                    text = f.read()

                pmid = article["pmid"]

                yield pmid, {
                    "text": text,  # articles[pmid],
                    "document_id": pmid,
                    "id": str(pmid),
                    "pmid": pmid,
                    "pmcid": article["pmcid"],
                    "grantList": [{"agency": grant["agency"]} for grant in article["grantList"]],
                }

        elif self.config.schema == "bigbio_text":
            for i, article in enumerate(task_data["articles"]):
                with open(filespath + "/" + article["pmcid"] + ".xml") as f:
                    text = f.read()

                yield article["pmid"], {
                    "text": text,
                    "id": str(article["pmid"]),
                    "document_id": article["pmid"],
                    "labels": [grant["agency"] for grant in article["grantList"]],
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
    """
    Fetches pubmed articles for a given list of PMIDs.

    PubMed articles can be downloaded in bulks, for now tested with 1000 articles per requests,
    but can still be slow. The BioASQ Task C 2017 contains up to 63 000 articles per split.

    Also required is a email address which is registered at https://pubmed.ncbi.nlm.nih.gov,
    we therefore discussed a extra attribute "cred_mail" in the BigBioConfig class, that can
    optionally be filled out. Additionally the dependecy `biopython` has to be introduced
    (https://biopython.org).

    For now this function also dumps all articles in arbitrary named json files in outdir.

    All the code on how to use this function/apporach is commented out in the data loader.
    """
    from Bio import Entrez

    # TODO: required to be set
    # Entrez.email = cred_mail

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    if not isinstance(pmids, list):
        pmids = list(pmids)

    n_chunks = int(len(pmids) / batch_size)

    for i in tqdm(range(0, n_chunks + (1 if n_chunks * batch_size < len(pmids) else 0))):
        start, end = i * batch_size, (i + 1) * batch_size
        outfname = f"{outdir}/pubmed.{i}.json"
        if os.path.exists(outfname) and not overwrite:
            continue

        query = ",".join(pmids[start:end])
        handle = Entrez.efetch(db="pubmed", id=query, rettype="gb", retmode="xml", retmax=batch_size)
        record = Entrez.read(handle)
        if len(record["PubmedArticle"]) != len(pmids[start:end]) and verbose:
            print(f"Queried {len(pmids[start:end])}, returned {len(record['PubmedArticle'])}")

        time.sleep(delay)
        # dump to JSON
        with open(outfname, "wt") as file:
            file.write(json.dumps(record, indent=2))
