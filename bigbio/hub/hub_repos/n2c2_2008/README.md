
---
language: 
- en
bigbio_language: 
- English
license: other
multilinguality: monolingual
bigbio_license_shortname: DUA
pretty_name: n2c2 2008 Obesity
homepage: https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/
bigbio_pubmed: True
bigbio_public: False
bigbio_tasks: 
- TEXT_CLASSIFICATION
---


# Dataset Card for n2c2 2008 Obesity

## Dataset Description

- **Homepage:** https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/
- **Pubmed:** True
- **Public:** False
- **Tasks:** TXTCLASS


The data for the n2c2 2008 obesity challenge consisted of discharge summaries from
the Partners HealthCare Research Patient Data Repository. These data were chosen 
from the discharge summaries of patients who were overweight or diabetic and had 
been hospitalized for obesity or diabetes sometime since 12/1/04. De-identification
was performed semi-automatically. All private health information was replaced with
synthetic identifiers.

The data for the challenge were annotated by two obesity experts from the 
Massachusetts General Hospital Weight Center. The experts were given a textual task, 
which asked them to classify each disease (see list of diseases above) as Present, 
Absent, Questionable, or Unmentioned based on explicitly documented information in 
the discharge summaries, e.g., the statement “the patient is obese”. The experts were 
also given an intuitive task, which asked them to classify each disease as Present, 
Absent, or Questionable by applying their intuition and judgment to information in 
the discharge summaries.



## Citation Information

```
@article{uzuner2009recognizing,
    author = {
        Uzuner, Ozlem
    },
    title     = {Recognizing Obesity and Comorbidities in Sparse Data},
    journal   = {Journal of the American Medical Informatics Association},
    volume    = {16},
    number    = {4},
    pages     = {561-570},
    year      = {2009},
    month     = {07},
    url       = {https://doi.org/10.1197/jamia.M3115},
    doi       = {10.1197/jamia.M3115},
    eprint    = {https://academic.oup.com/jamia/article-pdf/16/4/561/2302602/16-4-561.pdf}
}

```
