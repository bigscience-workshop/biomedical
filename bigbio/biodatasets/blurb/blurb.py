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
BLURB is a collection of resources for biomedical natural language processing. 
In general domains, such as newswire and the Web, comprehensive benchmarks and 
leaderboards such as GLUE have greatly accelerated progress in open-domain NLP. 
In biomedicine, however, such resources are ostensibly scarce. In the past, 
there have been a plethora of shared tasks in biomedical NLP, such as 
BioCreative, BioNLP Shared Tasks, SemEval, and BioASQ, to name just a few. These 
efforts have played a significant role in fueling interest and progress by the 
research community, but they typically focus on individual tasks. The advent of 
neural language models, such as BERT provides a unifying foundation to leverage 
transfer learning from unlabeled text to support a wide range of NLP 
applications. To accelerate progress in biomedical pretraining strategies and 
task-specific methods, it is thus imperative to create a broad-coverage 
benchmark encompassing diverse biomedical tasks. 

Inspired by prior efforts toward this direction (e.g., BLUE), we have created 
BLURB (short for Biomedical Language Understanding and Reasoning Benchmark). 
BLURB comprises of a comprehensive benchmark for PubMed-based biomedical NLP 
applications, as well as a leaderboard for tracking progress by the community. 
BLURB includes thirteen publicly available datasets in six diverse tasks. To 
avoid placing undue emphasis on tasks with many available datasets, such as 
named entity recognition (NER), BLURB reports the macro average across all tasks 
as the main score. The BLURB leaderboard is model-agnostic. Any system capable 
of producing the test predictions using the same training and development data 
can participate. The main goal of BLURB is to lower the entry barrier in 
biomedical NLP and help accelerate progress in this vitally important field for 
positive societal and human impact."""
import datasets
import pandas as pd
import re

from bigbio.utils import schemas
from bigbio.utils.configs import BigBioConfig
from bigbio.utils.constants import Lang, Tasks
from bigbio.utils.license import Licenses

_DATASETNAME = "blurb"
_DISPLAYNAME = "BLURB"

_LANGUAGES = [Lang.EN]
_PUBMED = True
_LOCAL = False
_CITATION = """\
@article{gu2021domain,
	title = {
		Domain-specific language model pretraining for biomedical natural
		language processing
	},
	author = {
		Gu, Yu and Tinn, Robert and Cheng, Hao and Lucas, Michael and
		Usuyama, Naoto and Liu, Xiaodong and Naumann, Tristan and Gao,
		Jianfeng and Poon, Hoifung
	},
	year = 2021,
	journal = {ACM Transactions on Computing for Healthcare (HEALTH)},
	publisher = {ACM New York, NY},
	volume = 3,
	number = 1,
	pages = {1--23}
}
"""


_BC2GM_DESCRIPTION = """\
The BioCreative II Gene Mention task. The training corpus for the current task \
consists mainly of the training and testing corpora (text collections) from the \
BCI task, and the testing corpus for the current task consists of an additional \
5,000 sentences that were held 'in reserve' from the previous task. In the \
current corpus, tokenization is not provided; instead participants are asked to \
identify a gene mention in a sentence by giving its start and end characters. As \
before, the training set consists of a set of sentences, and for each sentence a \
set of gene mentions (GENE annotations).

- Homepage: https://biocreative.bioinformatics.udel.edu/tasks/biocreative-ii/task-1a-gene-mention-tagging/
- Repository: https://github.com/cambridgeltl/MTL-Bioinformatics-2016/raw/master/data/
- Paper: Overview of BioCreative II gene mention recognition
         https://link.springer.com/article/10.1186/gb-2008-9-s2-s2
"""

_BC5_CHEM_DESCRIPTION = """\
The corpus consists of three separate sets of articles with diseases, chemicals \
and their relations annotated. The training (500 articles) and development (500 \
articles) sets were released to task participants in advance to support \
text-mining method development. The test set (500 articles) was used for final \
system performance evaluation.

- Homepage: https://biocreative.bioinformatics.udel.edu/resources/corpora/biocreative-v-cdr-corpus
- Repository: https://github.com/cambridgeltl/MTL-Bioinformatics-2016/raw/master/data/
- Paper: BioCreative V CDR task corpus: a resource for chemical disease relation extraction
         https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4860626/
"""

_BC5_DISEASE_DESCRIPTION = """\
The corpus consists of three separate sets of articles with diseases, chemicals \
and their relations annotated. The training (500 articles) and development (500 \
articles) sets were released to task participants in advance to support \
text-mining method development. The test set (500 articles) was used for final \
system performance evaluation.

- Homepage: https://biocreative.bioinformatics.udel.edu/resources/corpora/biocreative-v-cdr-corpus
- Repository: https://github.com/cambridgeltl/MTL-Bioinformatics-2016/raw/master/data/
- Paper: BioCreative V CDR task corpus: a resource for chemical disease relation extraction
         https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4860626/
"""

_JNLPBA_DESCRIPTION = """\
The BioNLP / JNLPBA Shared Task 2004 involves the identification and classification \
of technical terms referring to concepts of interest to biologists in the domain of \
molecular biology. The task was organized by GENIA Project based on the annotations \
of the GENIA Term corpus (version 3.02).

- Homepage: http://www.geniaproject.org/shared-tasks/bionlp-jnlpba-shared-task-2004
- Repository: https://github.com/cambridgeltl/MTL-Bioinformatics-2016/raw/master/data/
- Paper: Introduction to the Bio-entity Recognition Task at JNLPBA
         https://aclanthology.org/W04-1213
"""

_NCBI_DISEASE_DESCRIPTION = """\
[T]he NCBI disease corpus contains 6,892 disease mentions, which are mapped to \
790 unique disease concepts. Of these, 88% link to a MeSH identifier, while the \
rest contain an OMIM identifier. We were able to link 91% of the mentions to a \
single disease concept, while the rest are described as a combination of \
concepts.

- Homepage: https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/
- Repository: https://github.com/cambridgeltl/MTL-Bioinformatics-2016/raw/master/data/
- Paper: NCBI disease corpus: a resource for disease name recognition and concept normalization
         https://pubmed.ncbi.nlm.nih.gov/24393765/
"""

_EBM_PICO_DESCRIPTION = """"""

_CHEMPROT_DESCRIPTION = """"""
_DDI_DESCRIPTION = """"""
_GAD_DESCRIPTION = """"""

_BIOSSES_DESCRIPTION = """"""

_HOC_DESCRIPTION = """"""

_PUBMEDQA_DESCRIPTION = """"""
_BIOASQ_DESCRIPTION = """"""

_DESCRIPTION = {
    "bc2gm": _BC2GM_DESCRIPTION,
    "bc5disease": _BC5_DISEASE_DESCRIPTION,
    "bc5chem": _BC5_CHEM_DESCRIPTION,
    "jnlpba": _JNLPBA_DESCRIPTION,
    "ncbi_disease": _NCBI_DISEASE_DESCRIPTION,
}

_HOMEPAGE = "https://microsoft.github.io/BLURB/tasks.html"

_LICENSE = "MIXED"  # Licenses.GPL_3p0


_URLs = {
    "bc2gm": [
        "https://raw.githubusercontent.com/cambridgeltl/MTL-Bioinformatics-2016/master/data/BC2GM-IOB/train.tsv",
        "https://raw.githubusercontent.com/cambridgeltl/MTL-Bioinformatics-2016/master/data/BC2GM-IOB/devel.tsv",
        "https://raw.githubusercontent.com/cambridgeltl/MTL-Bioinformatics-2016/master/data/BC2GM-IOB/test.tsv",
    ],
    "bc5disease": [
        "https://raw.githubusercontent.com/cambridgeltl/MTL-Bioinformatics-2016/master/data/BC5CDR-disease-IOB/train.tsv",
        "https://raw.githubusercontent.com/cambridgeltl/MTL-Bioinformatics-2016/master/data/BC5CDR-disease-IOB/devel.tsv",
        "https://raw.githubusercontent.com/cambridgeltl/MTL-Bioinformatics-2016/master/data/BC5CDR-disease-IOB/test.tsv",
    ],
    "bc5chem": [
        "https://raw.githubusercontent.com/cambridgeltl/MTL-Bioinformatics-2016/master/data/BC5CDR-chem-IOB/train.tsv",
        "https://raw.githubusercontent.com/cambridgeltl/MTL-Bioinformatics-2016/master/data/BC5CDR-chem-IOB/devel.tsv",
        "https://raw.githubusercontent.com/cambridgeltl/MTL-Bioinformatics-2016/master/data/BC5CDR-chem-IOB/test.tsv",
    ],
    "jnlpba": [
        "https://raw.githubusercontent.com/cambridgeltl/MTL-Bioinformatics-2016/master/data/JNLPBA/train.tsv",
        "https://raw.githubusercontent.com/cambridgeltl/MTL-Bioinformatics-2016/master/data/JNLPBA/devel.tsv",
        "https://raw.githubusercontent.com/cambridgeltl/MTL-Bioinformatics-2016/master/data/JNLPBA/test.tsv",
    ],
    "ncbi_disease": [
        "https://raw.githubusercontent.com/cambridgeltl/MTL-Bioinformatics-2016/master/data/NCBI-disease-IOB/train.tsv",
        "https://raw.githubusercontent.com/cambridgeltl/MTL-Bioinformatics-2016/master/data/NCBI-disease-IOB/devel.tsv",
        "https://raw.githubusercontent.com/cambridgeltl/MTL-Bioinformatics-2016/master/data/NCBI-disease-IOB/test.tsv",
    ],
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION]
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class BlurbDataset(datasets.GeneratorBasedBuilder):
    """BIOSSES : Biomedical Semantic Similarity Estimation System"""

    DEFAULT_CONFIG_NAME = "biosses_source"
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="bc5chem",
            version=SOURCE_VERSION,
            description="BC5CDR Chemical IO Tagging",
            schema="ner",
            subset_id="bc5chem",
        ),
        BigBioConfig(
            name="bc5disease",
            version=SOURCE_VERSION,
            description="BC5CDR Chemical IO Tagging",
            schema="ner",
            subset_id="bc5disease",
        ),
        BigBioConfig(
            name="bc2gm",
            version=SOURCE_VERSION,
            description="BC2 Gene IO Tagging",
            schema="ner",
            subset_id="bc2gm",
        ),
        BigBioConfig(
            name="jnlpba",
            version=SOURCE_VERSION,
            description="JNLPBA Protein, DNA, RNA, Cell Type, Cell Line IO Tagging",
            schema="ner",
            subset_id="jnlpba",
        ),
        BigBioConfig(
            name="ncbi_disease",
            version=SOURCE_VERSION,
            description="NCBI Disease IO Tagging",
            schema="ner",
            subset_id="ncbi_disease",
        ),
    ]

    def _info(self):

        ner_features = datasets.Features(
            {
                "id": datasets.Value("string"),
                "tokens": datasets.Sequence(datasets.Value("string")),
                "word_idx": datasets.Value("int32"),
                "type": datasets.Value("string"),
                "tags": datasets.Sequence(
                    datasets.features.ClassLabel(
                        names=[
                            "O",
                            "I",
                        ]
                    )
                ),
            }
        )
        if self.config.schema == "ner":
            return datasets.DatasetInfo(
                description=_DESCRIPTION[self.config.name],
                features=ner_features,
                supervised_keys=None,
                homepage=_HOMEPAGE,
                license=str(_LICENSE),
                citation=_CITATION,
            )

    def _split_generators(self, dl_manager):

        my_urls = _URLs[self.config.name]
        dl_dir = dl_manager.download_and_extract(my_urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": dl_dir[0],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": dl_dir[1],
                    "split": "validation",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": dl_dir[2],
                    "split": "test",
                },
            ),
        ]

    def _load_iob(self, fpath):
        """
        Assumes input CoNLL file is a single entity type.
        """
        with open(fpath, "r") as file:
            tagged = []
            for line in file:
                if line.strip() == "":
                    toks, tags = zip(*tagged)
                    # transform tags
                    tags = [t[0] if t[0] in ["I", "O"] else "I" for t in tags]
                    yield (toks, tags)
                    tagged = []
                    continue
                tagged.append(re.split("\s", line.strip()))

            if tagged:
                toks, tags = zip(*tagged)
                tags = [t[0] for t in tags]
                yield (toks, tags)

    def _generate_examples(self, filepath, split):

        if self.config.schema == "ner":

            # Types for each NER dataset. Note BLURB's JNLPBA collapses all mentions into a
            # single entity type, which creates some ambiguity for prompting based on type
            ner_types = {
                "bc2gm": "gene",
                "bc5chem": "chemical",
                "bc5disease": "disease",
                "jnlpba": "protein, DNA, RNA, cell line, or cell type",
                "ncbi_disease": "disease",
            }

            uid = 0
            for item in self._load_iob(filepath):
                if len(item) != 2:
                    print("ERROR", len(item), item, split)
                    continue
                toks, tags = item
                for word_idx in range(len(toks)):
                    yield uid, {
                        "id": uid,
                        "tokens": toks,
                        "type": ner_types[self.config.name],
                        "tags": tags,
                        "word_idx": word_idx,
                    }
                    uid += 1
