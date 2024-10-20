
---
language: 
- en
bigbio_language: 
- English
license: other
multilinguality: monolingual
bigbio_license_shortname: DUA
pretty_name: n2c2 2006 Smoking Status
homepage: https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/
bigbio_pubmed: False
bigbio_public: False
bigbio_tasks: 
- TEXT_CLASSIFICATION
---


# Dataset Card for n2c2 2006 Smoking Status

## Dataset Description

- **Homepage:** https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/
- **Pubmed:** False
- **Public:** False
- **Tasks:** TXTCLASS


The data for the n2c2 2006 smoking challenge consisted of discharge summaries
from Partners HealthCare, which were then de-identified, tokenized, broken into
sentences, converted into XML format, and separated into training and test sets.

Two pulmonologists annotated each record with the smoking status of patients based
strictly on the explicitly stated smoking-related facts in the records. These
annotations constitute the textual judgments of the annotators. The annotators
were asked to classify patient records into five possible smoking status categories:
a past smoker, a current smoker, a smoker, a non-smoker and an unknown. A total of
502 de-identified medical discharge records were used for the smoking challenge.



## Citation Information

```
@article{uzuner2008identifying,
    author = {
        Uzuner, Ozlem and
        Goldstein, Ira and
        Luo, Yuan and
        Kohane, Isaac
    },
    title     = {Identifying Patient Smoking Status from Medical Discharge Records},
    journal   = {Journal of the American Medical Informatics Association},
    volume    = {15},
    number    = {1},
    pages     = {14-24},
    year      = {2008},
    month     = {01},
    url       = {https://doi.org/10.1197/jamia.M2408},
    doi       = {10.1136/amiajnl-2011-000784},
    eprint    = {https://academic.oup.com/jamia/article-pdf/15/1/14/2339646/15-1-14.pdf}
}

```
