
---
language: 
- fr
bigbio_language: 
- French
license: other
multilinguality: monolingual
bigbio_license_shortname: DUA
pretty_name: CAS
homepage: https://clementdalloux.fr/?page_id=28
bigbio_pubmed: False
bigbio_public: False
bigbio_tasks: 
- TEXT_CLASSIFICATION
---


# Dataset Card for CAS

## Dataset Description

- **Homepage:** https://clementdalloux.fr/?page_id=28
- **Pubmed:** False
- **Public:** False
- **Tasks:** TXTCLASS


We manually annotated two corpora from the biomedical field. The ESSAI corpus contains clinical trial protocols in French. They were mainly obtained from the National Cancer Institute The typical protocol consists of two parts: the summary of the trial, which indicates the purpose of the trial and the methods applied; and a detailed description of the trial with the inclusion and exclusion criteria. The CAS corpus contains clinical cases published in scientific literature and training material. They are published in different journals from French-speaking countries (France, Belgium, Switzerland, Canada, African countries, tropical countries) and are related to various medical specialties (cardiology, urology, oncology, obstetrics, pulmonology, gastro-enterology). The purpose of clinical cases is to describe clinical situations of patients. Hence, their content is close to the content of clinical narratives (description of diagnoses, treatments or procedures, evolution, family history, expected audience, etc.). In clinical cases, the negation is frequently used for describing the patient signs, symptoms, and diagnosis. Speculation is present as well but less frequently.

This version only contain the annotated CAS corpus



## Citation Information

```
@inproceedings{grabar-etal-2018-cas,
  title        = {{CAS}: {F}rench Corpus with Clinical Cases},
  author       = {Grabar, Natalia  and Claveau, Vincent  and Dalloux, Cl{'e}ment},
  year         = 2018,
  month        = oct,
  booktitle    = {
    Proceedings of the Ninth International Workshop on Health Text Mining and
    Information Analysis
  },
  publisher    = {Association for Computational Linguistics},
  address      = {Brussels, Belgium},
  pages        = {122--128},
  doi          = {10.18653/v1/W18-5614},
  url          = {https://aclanthology.org/W18-5614},
  abstract     = {
    Textual corpora are extremely important for various NLP applications as
    they provide information necessary for creating, setting and testing these
    applications and the corresponding tools. They are also crucial for
    designing reliable methods and reproducible results. Yet, in some areas,
    such as the medical area, due to confidentiality or to ethical reasons, it
    is complicated and even impossible to access textual data representative of
    those produced in these areas. We propose the CAS corpus built with
    clinical cases, such as they are reported in the published scientific
    literature in French. We describe this corpus, currently containing over
    397,000 word occurrences, and the existing linguistic and semantic
    annotations.
  }
}
```
