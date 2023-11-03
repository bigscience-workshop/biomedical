---
language:
  - en 
bigbio_language:
  - English
license: Creative Commons Zero v1.0 Universal
bigbio_license_shortname: cc0-1.0
multilinguality: monolingual
pretty_name: CZI DRSM
homepage: https://github.com/chanzuckerberg/DRSM-corpus
bigbio_pubmed: false
bigbio_public: true
bigbio_tasks:
  - TXTCLASS
---

# Dataset Card for CZI DRSM

## Dataset Description

- **Homepage:** https://github.com/chanzuckerberg/DRSM-corpus
- **Pubmed:** False
- **Public:** True
- **Tasks:** TXTCLASS

Research Article document classification dataset based on aspects of disease research. Currently, the dataset consists of three subsets: 

(A) classifies title/abstracts of papers into most popular subtypes of clinical, basic, and translational papers (~20k papers); 

    - Clinical Characteristics, Disease Pathology, and Diagnosis:
        Text that describes (i) symptoms, signs, or ‘phenotype’ of a disease; 
        (ii) the effects of the disease on patient organs, tissues, or cells; 
        (iii)) the results of clinical tests that reveal pathology (including
        biomarkers); (iv) research that use this information to figure out
        a diagnosis.

    - Therapeutics in the clinic: 
        Text describing how treatments work in the clinic (but not in a clinical trial).

    - Disease mechanism: 

    - Patient-Based Therapeutics: 
        Text describing (i) Clinical trials (studies of therapeutic measures being 
        used on patients in a clinical trial); (ii) Post Marketing Drug Surveillance 
        (effects of a drug after approval in the general population or as part of 
        ‘standard healthcare’); (iii) Drug repurposing (how a drug that has been 
        approved for one use is being applied to a new disease).

(B) identifies whether a title/abstract of a paper describes substantive research into Quality of Life (~10k papers); 

    - [-1] - the paper is not a primary experimental study in rare disease

    - [0] - the study does not directly investigate quality of life

    - [1] - the study investigates qol but not as its primary contribution

    - [2] - the study's primary contribution centers on quality of life measures

(C) identifies if a paper is a natural history study (~10k papers). 

   - [-1] - the paper is not a primary experimental study in rare disease

    - [0] - the study is not directly investigating the natural history of a disease

    - [1] - the study includes some elements a natural history but not as its primary contribution

    - [2] - the study's primary contribution centers on observing the time course of a rare disease
    
These classifications are particularly relevant in rare disease research, a field that is generally understudied. 

This data was compiled through the use of a gamified curation approach based on CentaurLabs' 'diagnos.us' platform.

## Citation Information

```
# N/A
```
