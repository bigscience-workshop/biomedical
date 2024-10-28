
---
language: 
- en
bigbio_language: 
- English
license: unknown
multilinguality: monolingual
bigbio_license_shortname: UNKNOWN
pretty_name: MeQSum
homepage: https://github.com/abachaa/MeQSum
bigbio_pubmed: False
bigbio_public: True
bigbio_tasks: 
- SUMMARIZATION
---


# Dataset Card for MeQSum

## Dataset Description

- **Homepage:** https://github.com/abachaa/MeQSum
- **Pubmed:** False
- **Public:** True
- **Tasks:** SUM


Dataset for medical question summarization introduced in the ACL 2019 paper "On the Summarization of Consumer Health
Questions". Question understanding is one of the main challenges in question answering. In real world applications,
users often submit natural language questions that are longer than needed and include peripheral information that
increases the complexity of the question, leading to substantially more false positives in answer retrieval. In this
paper, we study neural abstractive models for medical question summarization. We introduce the MeQSum corpus of 1,000
summarized consumer health questions.



## Citation Information

```
@inproceedings{ben-abacha-demner-fushman-2019-summarization,
    title = "On the Summarization of Consumer Health Questions",
    author = "Ben Abacha, Asma  and
      Demner-Fushman, Dina",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-1215",
    doi = "10.18653/v1/P19-1215",
    pages = "2228--2234",
    abstract = "Question understanding is one of the main challenges in question answering. In real world applications, users often submit natural language questions that are longer than needed and include peripheral information that increases the complexity of the question, leading to substantially more false positives in answer retrieval. In this paper, we study neural abstractive models for medical question summarization. We introduce the MeQSum corpus of 1,000 summarized consumer health questions. We explore data augmentation methods and evaluate state-of-the-art neural abstractive models on this new task. In particular, we show that semantic augmentation from question datasets improves the overall performance, and that pointer-generator networks outperform sequence-to-sequence attentional models on this task, with a ROUGE-1 score of 44.16{\%}. We also present a detailed error analysis and discuss directions for improvement that are specific to question summarization.",
}

```
