---
language:
  - en
bigbio_language:
  - English
license: apache-2.0
bigbio_license_shortname: APACHE_2p0
multilinguality: monolingual
pretty_name: Paragraph-level Simplification of Medical Texts
homepage: https://github.com/AshOlogn/Paragraph-level-Simplification-of-Medical-Texts
bigbio_pubmed: false
bigbio_public: true
bigbio_tasks:
  - SUM
paperswithcode_id: paragraph-level-simplification-of-medical
---


# Dataset Card for Paragraph-level Simplification of Medical Texts

## Dataset Description

- **Homepage:** https://github.com/AshOlogn/Paragraph-level-Simplification-of-Medical-Texts
- **Pubmed:** False
- **Public:** True
- **Tasks:** SUM


This dataset is designed for the summarization NLP task. It is a
collection of technical abstracts of biomedical systematic reviews
and corresponding plain-language summaries (PLS) from the Cochrane
Database of Systematic Reviews, which comprises thousands of evidence
synopses (where authors provide an overview of all published evidence
relevant to a particular clinical question or topic). The PLS are
written by review authors; Cochrane’s PLS standards recommend that
“the PLS should be written in plain English which can be understood by
most readers without a university education”. PLS are not parallel with
every sentence in the abstract; on the contrary, they are structured heterogeneously.


## Citation Information

```
@inproceedings{devaraj-etal-2021-paragraph,
    title = "Paragraph-level Simplification of Medical Texts",
    author = "Devaraj, Ashwin and Marshall, Iain and Wallace, Byron and Li, Junyi Jessy",
    booktitle = {Proceedings of the 2021 Conference of the North
                American Chapter of the Association for Computational Linguistics},
    month = jun,
    year = "2021",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.395",
    pages = "4972--4984",
}
```