---
language: 
  - en
bigbio_language:
  - English
license: other
multilinguality: monolingual
bigbio_license_shortname: MIXED
pretty_name: BLURB
homepage: https://microsoft.github.io/BLURB/tasks.html
bigbio_pubmed: true
bigbio_public: true
bigbio_tasks:
  - NAMED_ENTITY_RECOGNITION
---


# Dataset Card for BLURB

## Dataset Description

- **Homepage:** https://microsoft.github.io/BLURB/tasks.html
- **Pubmed:** True
- **Public:** True
- **Tasks:** NER

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
positive societal and human impact.

This implementation contains a subset of 5 tasks as of 2022.10.06, with their original train, dev, and test splits.


## Citation Information

```
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
```
