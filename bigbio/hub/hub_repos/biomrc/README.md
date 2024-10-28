
---
language: 
- en
bigbio_language: 
- English
license: unknown
multilinguality: monolingual
bigbio_license_shortname: UNKNOWN
pretty_name: BIOMRC
homepage: https://github.com/PetrosStav/BioMRC_code
bigbio_pubmed: True
bigbio_public: True
bigbio_tasks: 
- QUESTION_ANSWERING
---


# Dataset Card for BIOMRC

## Dataset Description

- **Homepage:** https://github.com/PetrosStav/BioMRC_code
- **Pubmed:** True
- **Public:** True
- **Tasks:** QA


We introduce BIOMRC, a large-scale cloze-style biomedical MRC dataset. Care was taken to reduce noise, compared to the
previous BIOREAD dataset of Pappas et al. (2018). Experiments show that simple heuristics do not perform well on the
new dataset and that two neural MRC models that had been tested on BIOREAD perform much better on BIOMRC, indicating
that the new dataset is indeed less noisy or at least that its task is more feasible. Non-expert human performance is
also higher on the new dataset compared to BIOREAD, and biomedical experts perform even better. We also introduce a new
BERT-based MRC model, the best version of which substantially outperforms all other methods tested, reaching or
surpassing the accuracy of biomedical experts in some experiments. We make the new dataset available in three different
sizes, also releasing our code, and providing a leaderboard.



## Citation Information

```
@inproceedings{pappas-etal-2020-biomrc,
    title = "{B}io{MRC}: A Dataset for Biomedical Machine Reading Comprehension",
    author = "Pappas, Dimitris  and
      Stavropoulos, Petros  and
      Androutsopoulos, Ion  and
      McDonald, Ryan",
    booktitle = "Proceedings of the 19th SIGBioMed Workshop on Biomedical Language Processing",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.bionlp-1.15",
    pages = "140--149",
}

```
