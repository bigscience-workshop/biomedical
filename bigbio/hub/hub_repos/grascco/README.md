---
language:
  - de
bigbio_language:
  - German
license: cc-by-4.0
bigbio_license_shortname: CC_BY_4p0
multilinguality: monolingual
pretty_name: GraSCCo
homepage: https://zenodo.org/records/6539131
bigbio_pubmed: false
bigbio_public: true
bigbio_tasks:
  - NAMED_ENTITY_RECOGNITION
---


# Dataset Card for GraSCCo

## Dataset Description

- **Homepage:** https://zenodo.org/records/6539131
- **Pubmed:** False
- **Public:** True
- **Tasks:** NER

GraSCCo is a collection of artificially generated semi-structured and unstructured German-language clinical summaries. These summaries are formulated as letters from the hospital to the patient's GP after in-patient or out-patient care.
This is common practice in Germany, Austria and Switzerland.

The creation of the GraSCCo documents were inspired by existing clinical texts, but all names and dates are purely fictional.
There is no relation to existing patients, clinicians or institutions. Whereas the texts try to represent the range of German clinical language as best as possible, medical plausibility must not be assumed.

GraSCCo can therefore only be used to train clinical language models, not clinical domain models. 


## Citation Information

```
@incollection{modersohn2022grascco,
  title={GRASCCO—The First Publicly Shareable, Multiply-Alienated German Clinical Text Corpus},
  author={Modersohn, Luise and Schulz, Stefan and Lohr, Christina and Hahn, Udo},
  booktitle={German Medical Data Sciences 2022--Future Medicine: More Precise, More Integrative, More Sustainable!},
  pages={66--72},
  year={2022},
  publisher={IOS Press}
}
```
