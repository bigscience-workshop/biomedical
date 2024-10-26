
---
language: 
- es
bigbio_language: 
- Spanish
license: cc-by-4.0
multilinguality: monolingual
bigbio_license_shortname: CC_BY_4p0
pretty_name: CANTEMIST
homepage: https://temu.bsc.es/cantemist/?p=4338
bigbio_pubmed: False
bigbio_public: True
bigbio_tasks: 
- NAMED_ENTITY_RECOGNITION
- NAMED_ENTITY_DISAMBIGUATION
- TEXT_CLASSIFICATION
---


# Dataset Card for CANTEMIST

## Dataset Description

- **Homepage:** https://temu.bsc.es/cantemist/?p=4338
- **Pubmed:** False
- **Public:** True
- **Tasks:** NER,NED,TXTCLASS


Collection of 1301 oncological clinical case reports written in Spanish, with tumor morphology mentions manually annotated and mapped by clinical experts to a controlled terminology. Every tumor morphology mention is linked to an eCIE-O code (the Spanish equivalent of ICD-O).

The original dataset is distributed in Brat format, and was randomly sampled into 3 subsets. The training, development and test sets contain 501, 500 and 300 documents each, respectively.

This dataset was designed for the CANcer TExt Mining Shared Task, sponsored by Plan-TL. The task is divided in 3 subtasks: CANTEMIST-NER, CANTEMIST_NORM and CANTEMIST-CODING.

CANTEMIST-NER track: requires finding automatically tumor morphology mentions. All tumor morphology mentions are defined by their corresponding character offsets in UTF-8 plain text medical documents. 

CANTEMIST-NORM track: clinical concept normalization or named entity normalization task that requires to return all tumor morphology entity mentions together with their corresponding eCIE-O-3.1 codes i.e. finding and normalizing tumor morphology mentions.

CANTEMIST-CODING track: requires returning for each of document a ranked list of its corresponding ICD-O-3 codes. This it is essentially a sort of indexing or multi-label classification task or oncology clinical coding. 

For further information, please visit https://temu.bsc.es/cantemist or send an email to encargo-pln-life@bsc.es



## Citation Information

```
@article{miranda2020named,
  title={Named Entity Recognition, Concept Normalization and Clinical Coding: Overview of the Cantemist Track for Cancer Text Mining in Spanish, Corpus, Guidelines, Methods and Results.},
  author={Miranda-Escalada, Antonio and Farr{'e}, Eul{\`a}lia and Krallinger, Martin},
  journal={IberLEF@ SEPLN},
  pages={303--323},
  year={2020}
}

```
