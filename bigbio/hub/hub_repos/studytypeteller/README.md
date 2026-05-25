---
language:
  - en
bigbio_language:
  - English
license: apache-2.0
bigbio_license_shortname: APACHE_2p0
multilinguality: monolingual
pretty_name: StudyTypeTeller
homepage: https://github.com/Ineichen-Group/StudyTypeTeller/tree/main
bigbio_pubmed: true
bigbio_public: true
bigbio_tasks:
  - TEXT_CLASSIFICATION
---


# Dataset Card for StudyTypeTeller

## Dataset Description

- **Homepage:** https://github.com/Ineichen-Group/StudyTypeTeller/tree/main
- **Pubmed:** True
- **Public:** True
- **Tasks:** TXTCLASS


The dataset consists of 2,645 PubMed titles and abstracts, manually annotated for study type classification in the biomedical domain. 
This corpus is designed to support automated classification of study types and to facilitate research on reducing the workload associated with abstract screening in systematic reviews.

Each record is labeled according to one of 14 predefined study types.
The annotations were created by human experts, and include the following categories:

| Category                                      | Train | Validation | Test | Total |
|----------------------------------------------|------:|-----------:|-----:|------:|
| **Corpus size**                              | 1,581 |        530 |  534 | 2,645 |
| **Multiclass labels**                        |       |            |      |       |
| Remaining                                    |   514 |        172 |  172 |   858 |
| Non-systematic-review                        |   222 |         74 |   75 |   371 |
| Human-non-RCT-non-drug-intervention          |   123 |         41 |   42 |   206 |
| Human-non-RCT-drug-intervention              |   107 |         36 |   36 |   179 |
| Human-case-report                            |    99 |         33 |   33 |   165 |
| Animal-other                                 |    93 |         31 |   31 |   155 |
| Animal-drug-intervention                     |    88 |         29 |   30 |   147 |
| Human-systematic-review                      |    66 |         22 |   22 |   110 |
| In-vitro-study                               |    62 |         21 |   21 |   104 |
| Human-RCT-non-drug-intervention              |    52 |         18 |   18 |    88 |
| Animal-non-drug-intervention                 |    46 |         15 |   16 |    77 |
| Human-RCT-drug-intervention                  |    43 |         15 |   15 |    73 |
| Clinical-study-protocol                      |    34 |         12 |   12 |    58 |
| Human-RCT-non-intervention                   |    32 |         11 |   11 |    54 |

## Citation Information

```
@article{doneva2025studytypeteller,
  author    = {Doneva, Emilova S and de Viragh, S and Hubarava, H and Schandelmaier, S and Briel, M and Ineichen, BV},
  title     = {StudyTypeTeller—Large language models to automatically classify research study types for systematic reviews},
  journal   = {Research Synthesis Methods},
  volume    = {16},
  number    = {6},
  pages     = {1005--1024},
  year      = {2025},
  doi       = {10.1017/rsm.2025.10031}
}
```
