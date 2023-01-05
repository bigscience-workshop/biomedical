
---
language: 
- en
bigbio_language: 
- English
license: cc-by-sa-3.0
multilinguality: monolingual
bigbio_license_shortname: CC_BY_SA_3p0
pretty_name: MedHop
homepage: http://qangaroo.cs.ucl.ac.uk/
bigbio_pubmed: True
bigbio_public: True
bigbio_tasks: 
- QUESTION_ANSWERING
---


# Dataset Card for MedHop

## Dataset Description

- **Homepage:** http://qangaroo.cs.ucl.ac.uk/
- **Pubmed:** True
- **Public:** True
- **Tasks:** QA


With the same format as WikiHop, this dataset is based on research paper
abstracts from PubMed, and the queries are about interactions between
pairs of drugs. The correct answer has to be inferred by combining
information from a chain of reactions of drugs and proteins.



## Citation Information

```
@article{welbl-etal-2018-constructing,
title = Constructing Datasets for Multi-hop Reading Comprehension Across Documents,
author = Welbl, Johannes and Stenetorp, Pontus and Riedel, Sebastian,
journal = Transactions of the Association for Computational Linguistics,
volume = 6,
year = 2018,
address = Cambridge, MA,
publisher = MIT Press,
url = https://aclanthology.org/Q18-1021,
doi = 10.1162/tacl_a_00021,
pages = 287--302,
abstract = {
    Most Reading Comprehension methods limit themselves to queries which
    can be answered using a single sentence, paragraph, or document.
    Enabling models to combine disjoint pieces of textual evidence would
    extend the scope of machine comprehension methods, but currently no
    resources exist to train and test this capability. We propose a novel
    task to encourage the development of models for text understanding
    across multiple documents and to investigate the limits of existing
    methods. In our task, a model learns to seek and combine evidence
    -- effectively performing multihop, alias multi-step, inference.
    We devise a methodology to produce datasets for this task, given a
    collection of query-answer pairs and thematically linked documents.
    Two datasets from different domains are induced, and we identify
    potential pitfalls and devise circumvention strategies. We evaluate
    two previously proposed competitive models and find that one can
    integrate information across documents. However, both models
    struggle to select relevant information; and providing documents
    guaranteed to be relevant greatly improves their performance. While
    the models outperform several strong baselines, their best accuracy
    reaches 54.5 % on an annotated test set, compared to human
    performance at 85.0 %, leaving ample room for improvement.
}

```
