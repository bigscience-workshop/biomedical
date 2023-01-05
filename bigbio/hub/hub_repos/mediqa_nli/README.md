
---
language: 
- en
bigbio_language: 
- English
license: other
multilinguality: monolingual
bigbio_license_shortname: PHYSIONET_LICENSE_1p5
pretty_name: MEDIQA NLI
homepage: https://physionet.org/content/mednli-bionlp19/1.0.1/
bigbio_pubmed: False
bigbio_public: False
bigbio_tasks: 
- TEXTUAL_ENTAILMENT
---


# Dataset Card for MEDIQA NLI

## Dataset Description

- **Homepage:** https://physionet.org/content/mednli-bionlp19/1.0.1/
- **Pubmed:** False
- **Public:** False
- **Tasks:** TE


Natural Language Inference (NLI) is the task of determining whether a given hypothesis can be
inferred from a given premise. Also known as Recognizing Textual Entailment (RTE), this task has
enjoyed popularity among researchers for some time. However, almost all datasets for this task
focused on open domain data such as as news texts, blogs, and so on. To address this gap, the MedNLI
dataset was created for language inference in the medical domain. MedNLI is a derived dataset with
data sourced from MIMIC-III v1.4. In order to stimulate research for this problem, a shared task on
Medical Inference and Question Answering (MEDIQA) was organized at the workshop for biomedical
natural language processing (BioNLP) 2019. The dataset provided herein is a test set of 405 premise
hypothesis pairs for the NLI challenge in the MEDIQA shared task. Participants of the shared task
are expected to use the MedNLI data for development of their models and this dataset was used as an
unseen dataset for scoring each participant submission.



## Citation Information

```
@misc{https://doi.org/10.13026/gtv4-g455,
    title        = {MedNLI for Shared Task at ACL BioNLP 2019},
    author       = {Shivade,  Chaitanya},
    year         = 2019,
    publisher    = {physionet.org},
    doi          = {10.13026/GTV4-G455},
    url          = {https://physionet.org/content/mednli-bionlp19/}
}


```
