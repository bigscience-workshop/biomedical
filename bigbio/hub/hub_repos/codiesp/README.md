
---
language: 
- es
bigbio_language: 
- Spanish
license: cc-by-4.0
multilinguality: monolingual
bigbio_license_shortname: CC_BY_4p0
pretty_name: CodiEsp
homepage: https://temu.bsc.es/codiesp/
bigbio_pubmed: False
bigbio_public: True
bigbio_tasks: 
- TEXT_CLASSIFICATION
- NAMED_ENTITY_RECOGNITION
- NAMED_ENTITY_DISAMBIGUATION
---


# Dataset Card for CodiEsp

## Dataset Description

- **Homepage:** https://temu.bsc.es/codiesp/
- **Pubmed:** False
- **Public:** True
- **Tasks:** TXTCLASS,NER,NED


Synthetic corpus of 1,000 manually selected clinical case studies in Spanish
that was designed for the Clinical Case Coding in Spanish Shared Task, as part
of the CLEF 2020 conference.

The goal of the task was to automatically assign ICD10 codes (CIE-10, in
Spanish) to clinical case documents, being evaluated against manually generated
ICD10 codifications. The CodiEsp corpus was selected manually by practicing
physicians and clinical documentalists and annotated by clinical coding
professionals meeting strict quality criteria. They reached an inter-annotator
agreement of 88.6% for diagnosis coding, 88.9% for procedure coding and 80.5%
for the textual reference annotation.

The final collection of 1,000 clinical cases that make up the corpus had a total
of 16,504 sentences and 396,988 words. All documents are in Spanish language and
CIE10 is the coding terminology (the Spanish version of ICD10-CM and ICD10-PCS).
The CodiEsp corpus has been randomly sampled into three subsets. The train set
contains 500 clinical cases, while the development and test sets have 250
clinical cases each. In addition to these, a collection of 176,294 abstracts
from Lilacs and Ibecs with the corresponding ICD10 codes (ICD10-CM and
ICD10-PCS) was provided by the task organizers. Every abstract has at least one
associated code, with an average of 2.5 ICD10 codes per abstract.

The CodiEsp track was divided into three sub-tracks (2 main and 1 exploratory):

- CodiEsp-D: The Diagnosis Coding sub-task, which requires automatic ICD10-CM
  [CIE10-Diagnóstico] code assignment.
- CodiEsp-P: The Procedure Coding sub-task, which requires automatic ICD10-PCS
  [CIE10-Procedimiento] code assignment.
- CodiEsp-X: The Explainable AI exploratory sub-task, which requires to submit
  the reference to the predicted codes (both ICD10-CM and ICD10-PCS). The goal 
  of this novel task was not only to predict the correct codes but also to 
  present the reference in the text that supports the code predictions.

For further information, please visit https://temu.bsc.es/codiesp or send an
email to encargo-pln-life@bsc.es



## Citation Information

```
@article{miranda2020overview,
  title={Overview of Automatic Clinical Coding: Annotations, Guidelines, and Solutions for non-English Clinical Cases at CodiEsp Track of CLEF eHealth 2020.},
  author={Miranda-Escalada, Antonio and Gonzalez-Agirre, Aitor and Armengol-Estap{'e}, Jordi and Krallinger, Martin},
  journal={CLEF (Working Notes)},
  volume={2020},
  year={2020}
}

```
