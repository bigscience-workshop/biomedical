
---
language: 
- en
bigbio_language: 
- English
license: unknown
multilinguality: monolingual
bigbio_license_shortname: UNKNOWN
pretty_name: BiologyHowWhyCorpus
homepage: https://allenai.org/data/biology-how-why-corpus
bigbio_pubmed: False
bigbio_public: True
bigbio_tasks: 
- QUESTION_ANSWERING
---


# Dataset Card for BiologyHowWhyCorpus

## Dataset Description

- **Homepage:** https://allenai.org/data/biology-how-why-corpus
- **Pubmed:** False
- **Public:** True
- **Tasks:** QA


This dataset consists of 185 "how" and 193 "why" biology questions authored by a domain expert, with one or more gold 
answer passages identified in an undergraduate textbook. The expert was not constrained in any way during the 
annotation process, so gold answers might be smaller than a paragraph or span multiple paragraphs. This dataset was 
used for the question-answering system described in the paper “Discourse Complements Lexical Semantics for Non-factoid 
Answer Reranking” (ACL 2014).



## Citation Information

```
@inproceedings{jansen-etal-2014-discourse,
    title = "Discourse Complements Lexical Semantics for Non-factoid Answer Reranking",
    author = "Jansen, Peter  and
      Surdeanu, Mihai  and
      Clark, Peter",
    booktitle = "Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jun,
    year = "2014",
    address = "Baltimore, Maryland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P14-1092",
    doi = "10.3115/v1/P14-1092",
    pages = "977--986",
}

```
