
---
language: 
- en
bigbio_language: 
- English
license: cc0-1.0
multilinguality: monolingual
bigbio_license_shortname: CC0_1p0
pretty_name: NLM-Gene
homepage: https://zenodo.org/record/5089049
bigbio_pubmed: True
bigbio_public: True
bigbio_tasks: 
- NAMED_ENTITY_RECOGNITION
- NAMED_ENTITY_DISAMBIGUATION
---


# Dataset Card for NLM-Gene

## Dataset Description

- **Homepage:** https://zenodo.org/record/5089049
- **Pubmed:** True
- **Public:** True
- **Tasks:** NER,NED


NLM-Gene consists of 550 PubMed articles, from 156 journals, and contains more than 15 thousand unique gene names, corresponding to more than five thousand gene identifiers (NCBI Gene taxonomy). This corpus contains gene annotation data from 28 organisms. The annotated articles contain on average 29 gene names, and 10 gene identifiers per article. These characteristics demonstrate that this article set is an important benchmark dataset to test the accuracy of gene recognition algorithms both on multi-species and ambiguous data. The NLM-Gene corpus will be invaluable for advancing text-mining techniques for gene identification tasks in biomedical text.



## Citation Information

```
@article{islamaj2021nlm,
  title        = {
    NLM-Gene, a richly annotated gold standard dataset for gene entities that
    addresses ambiguity and multi-species gene recognition
  },
  author       = {
    Islamaj, Rezarta and Wei, Chih-Hsuan and Cissel, David and Miliaras,
    Nicholas and Printseva, Olga and Rodionov, Oleg and Sekiya, Keiko and Ward,
    Janice and Lu, Zhiyong
  },
  year         = 2021,
  journal      = {Journal of Biomedical Informatics},
  publisher    = {Elsevier},
  volume       = 118,
  pages        = 103779
}

```
