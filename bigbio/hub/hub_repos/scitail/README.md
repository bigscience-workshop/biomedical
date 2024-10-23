---
language: 
  - en
bigbio_language:
  - English
license: apache-2.0
bigbio_license_shortname: APACHE_2p0
multilinguality: monolingual
pretty_name: SciTail
homepage: https://allenai.org/data/scitail
bigbio_pubmed: false
bigbio_public: true
bigbio_tasks:
  - TEXTUAL_ENTAILMENT
paperswithcode_id: scitail
---


# Dataset Card for SciTail

## Dataset Description

- **Homepage:** https://allenai.org/data/scitail
- **Pubmed:** False
- **Public:** True
- **Tasks:** TE


The SciTail dataset is an entailment dataset created from multiple-choice science exams and web sentences. Each question and the correct answer choice are converted into an assertive statement to form the hypothesis. We use information retrieval to obtain relevant text from a large text corpus of web sentences, and use these sentences as a premise P. We crowd source the annotation of such premise-hypothesis pair as supports (entails) or not (neutral), in order to create the SciTail dataset. The dataset contains 27,026 examples with 10,101 examples with entails label and 16,925 examples with neutral label.


## Citation Information

```
@inproceedings{scitail,
    author = {Tushar Khot and Ashish Sabharwal and Peter Clark},
    booktitle = {AAAI}
    title = {SciTail: A Textual Entailment Dataset from Science Question Answering},
    year = {2018}
```
