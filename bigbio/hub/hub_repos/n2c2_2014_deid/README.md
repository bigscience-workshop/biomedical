
---
language: 
- en
bigbio_language: 
- English
license: other
multilinguality: monolingual
bigbio_license_shortname: DUA
pretty_name: n2c2 2014 De-identification
homepage: https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/
bigbio_pubmed: False
bigbio_public: False
bigbio_tasks: 
- NAMED_ENTITY_RECOGNITION
---


# Dataset Card for n2c2 2014 De-identification

## Dataset Description

- **Homepage:** https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/
- **Pubmed:** False
- **Public:** False
- **Tasks:** NER


The 2014 i2b2/UTHealth Natural Language Processing (NLP) shared task featured two tracks.
The first of these was the de-identification track focused on identifying protected health
information (PHI) in longitudinal clinical narratives.

TRACK 1: NER PHI

HIPAA requires that patient medical records have all identifying information removed in order to
protect patient privacy. There are 18 categories of Protected Health Information (PHI) identifiers of the
patient or of relatives, employers, or household members of the patient that must be removed in order
for a file to be considered de-identified.
In order to de-identify the records, each file has PHI marked up. All PHI has an
XML tag indicating its category and type, where applicable. For the purposes of this task,
the 18 HIPAA categories have been grouped into 6 main categories and 25 sub categories



## Citation Information

```
@article{stubbs2015automated,
title = {Automated systems for the de-identification of longitudinal
clinical narratives: Overview of 2014 i2b2/UTHealth shared task Track 1},
journal = {Journal of Biomedical Informatics},
volume = {58},
pages = {S11-S19},
year = {2015},
issn = {1532-0464},
doi = {https://doi.org/10.1016/j.jbi.2015.06.007},
url = {https://www.sciencedirect.com/science/article/pii/S1532046415001173},
author = {Amber Stubbs and Christopher Kotfila and Özlem Uzuner}
}

```
