
---
language: 
- en
bigbio_language: 
- English
license: unknown
multilinguality: monolingual
bigbio_license_shortname: UNKNOWN
pretty_name: BioRelEx
homepage: https://github.com/YerevaNN/BioRelEx
bigbio_pubmed: True
bigbio_public: True
bigbio_tasks: 
- NAMED_ENTITY_RECOGNITION
- NAMED_ENTITY_DISAMBIGUATION
- RELATION_EXTRACTION
- COREFERENCE_RESOLUTION
---


# Dataset Card for BioRelEx

## Dataset Description

- **Homepage:** https://github.com/YerevaNN/BioRelEx
- **Pubmed:** True
- **Public:** True
- **Tasks:** NER,NED,RE,COREF


BioRelEx is a biological relation extraction dataset. Version 1.0 contains 2010
annotated sentences that describe binding interactions between various
biological entities (proteins, chemicals, etc.). 1405 sentences are for
training, another 201 sentences are for validation. They are publicly available
at https://github.com/YerevaNN/BioRelEx/releases. Another 404 sentences are for
testing which are kept private for at this Codalab competition
https://competitions.codalab.org/competitions/20468. All sentences contain words
"bind", "bound" or "binding". For every sentence we provide: 1) Complete
annotations of all biological entities that appear in the sentence 2) Entity
types (32 types) and grounding information for most of the proteins and families
(links to uniprot, interpro and other databases) 3) Coreference between entities
in the same sentence (e.g. abbreviations and synonyms) 4) Binding interactions
between the annotated entities 5) Binding interaction types: positive, negative
(A does not bind B) and neutral (A may bind to B)


## Citation Information

```
@inproceedings{khachatrian2019biorelex,
    title = "{B}io{R}el{E}x 1.0: Biological Relation Extraction Benchmark",
    author = "Khachatrian, Hrant  and
      Nersisyan, Lilit  and
      Hambardzumyan, Karen  and
      Galstyan, Tigran  and
      Hakobyan, Anna  and
      Arakelyan, Arsen  and
      Rzhetsky, Andrey  and
      Galstyan, Aram",
    booktitle = "Proceedings of the 18th BioNLP Workshop and Shared Task",
    month = aug,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W19-5019",
    doi = "10.18653/v1/W19-5019",
    pages = "176--190"
}

```
