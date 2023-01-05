
---
language: 
- en
bigbio_language: 
- English
license: mit
multilinguality: monolingual
bigbio_license_shortname: MIT
pretty_name: PubMedQA
homepage: https://github.com/pubmedqa/pubmedqa
bigbio_pubmed: True
bigbio_public: True
bigbio_tasks: 
- QUESTION_ANSWERING
---


# Dataset Card for PubMedQA

## Dataset Description

- **Homepage:** https://github.com/pubmedqa/pubmedqa
- **Pubmed:** True
- **Public:** True
- **Tasks:** QA


PubMedQA is a novel biomedical question answering (QA) dataset collected from PubMed abstracts.
The task of PubMedQA is to answer research biomedical questions with yes/no/maybe using the corresponding abstracts.
PubMedQA has 1k expert-annotated (PQA-L), 61.2k unlabeled (PQA-U) and 211.3k artificially generated QA instances (PQA-A).

Each PubMedQA instance is composed of:
  (1) a question which is either an existing research article title or derived from one,
  (2) a context which is the corresponding PubMed abstract without its conclusion,
  (3) a long answer, which is the conclusion of the abstract and, presumably, answers the research question, and
  (4) a yes/no/maybe answer which summarizes the conclusion.

PubMedQA is the first QA dataset where reasoning over biomedical research texts,
especially their quantitative contents, is required to answer the questions.

PubMedQA datasets comprise of 3 different subsets:
  (1) PubMedQA Labeled (PQA-L): A labeled PubMedQA subset comprises of 1k manually annotated yes/no/maybe QA data collected from PubMed articles.
  (2) PubMedQA Artificial (PQA-A): An artificially labelled PubMedQA subset comprises of 211.3k PubMed articles with automatically generated questions from the statement titles and yes/no answer labels generated using a simple heuristic.
  (3) PubMedQA Unlabeled (PQA-U): An unlabeled PubMedQA subset comprises of 61.2k context-question pairs data collected from PubMed articles.



## Citation Information

```
@inproceedings{jin2019pubmedqa,
  title={PubMedQA: A Dataset for Biomedical Research Question Answering},
  author={Jin, Qiao and Dhingra, Bhuwan and Liu, Zhengping and Cohen, William and Lu, Xinghua},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages={2567--2577},
  year={2019}
}

```
