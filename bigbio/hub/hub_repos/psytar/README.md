
---
language: 
- en
bigbio_language: 
- English
license: cc-by-4.0
multilinguality: monolingual
bigbio_license_shortname: CC_BY_4p0
pretty_name: PsyTAR
homepage: https://www.askapatient.com/research/pharmacovigilance/corpus-ades-psychiatric-medications.asp
bigbio_pubmed: False
bigbio_public: False
bigbio_tasks: 
- NAMED_ENTITY_RECOGNITION
- TEXT_CLASSIFICATION
---


# Dataset Card for PsyTAR

## Dataset Description

- **Homepage:** https://www.askapatient.com/research/pharmacovigilance/corpus-ades-psychiatric-medications.asp
- **Pubmed:** False
- **Public:** False
- **Tasks:** NER,TXTCLASS


The "Psychiatric Treatment Adverse Reactions" (PsyTAR) dataset contains 891 drugs
reviews posted by patients on "askapatient.com", about the effectiveness and adverse
drug events associated with Zoloft, Lexapro, Cymbalta, and Effexor XR.

This dataset can be used for (multi-label) sentence classification of Adverse Drug
Reaction (ADR), Withdrawal Symptoms (WDs), Sign/Symptoms/Illness (SSIs), Drug
Indications (DIs), Drug Effectiveness (EF), Drug Infectiveness (INF) and Others, as well
as for recognition of 5 different types of named entity (in the categories ADRs, WDs,
SSIs and DIs)



## Citation Information

```
@article{Zolnoori2019,
  author    = {Maryam Zolnoori and
               Kin Wah Fung and
               Timothy B. Patrick and
               Paul Fontelo and
               Hadi Kharrazi and
               Anthony Faiola and
               Yi Shuan Shirley Wu and
               Christina E. Eldredge and
               Jake Luo and
               Mike Conway and
               Jiaxi Zhu and
               Soo Kyung Park and
               Kelly Xu and
               Hamideh Moayyed and
               Somaieh Goudarzvand},
  title     = {A systematic approach for developing a corpus of patient                reported adverse drug events: A case study for {SSRI} and {SNRI} medications},
  journal   = {Journal of Biomedical Informatics},
  volume    = {90},
  year      = {2019},
  url       = {https://doi.org/10.1016/j.jbi.2018.12.005},
  doi       = {10.1016/j.jbi.2018.12.005},
}

```
