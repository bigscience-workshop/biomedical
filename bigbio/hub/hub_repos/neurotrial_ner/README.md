---
language:
  - en
bigbio_language:
  - English
license: cc0-1.0
bigbio_license_shortname: CC0_1p0
multilinguality: monolingual
pretty_name: NeuroTrialNer
homepage: https://github.com/Ineichen-Group/NeuroTrialNER/tree/main
bigbio_pubmed: false
bigbio_public: true
bigbio_tasks:
  - NAMED_ENTITY_RECOGNITION
---


# Dataset Card for NeuroTrialNer

## Dataset Description

- **Homepage:** https://github.com/Ineichen-Group/NeuroTrialNER/tree/main
- **Pubmed:** False
- **Public:** True
- **Tasks:** NER


NeuoTrialNER is an annotated dataset for named entities in clinical trial registry data in the domain of neurology/psychiatry. 
The corpus comprises 1093 clinical trial title and brief summaries from ClinicalTrials.gov. 
It has been annotated by two to three annotators for key trial characteristics, i.e., condition (e.g., Alzheimer's disease), 
therapeutic intervention (e.g., aspirin), and control arms (e.g., placebo).

## Citation Information

```
@inproceedings{doneva-etal-2024-neurotrialner,
    title = "{N}euro{T}rial{NER}: An Annotated Corpus for Neurological Diseases and Therapies in Clinical Trial Registries",
    author = "Doneva, Simona Emilova  and
      Ellendorff, Tilia  and
      Sick, Beate  and
      Goldman, Jean-Philippe  and
      Cannon, Amelia Elaine  and
      Schneider, Gerold  and
      Ineichen, Benjamin Victor",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.1050",
    pages = "18868--18890",
    abstract = "Extracting and aggregating information from clinical trial registries could provide invaluable insights into the drug development landscape and advance the treatment of neurologic diseases. However, achieving this at scale is hampered by the volume of available data and the lack of an annotated corpus to assist in the development of automation tools. Thus, we introduce NeuroTrialNER, a new and fully open corpus for named entity recognition (NER). It comprises 1093 clinical trial summaries sourced from ClinicalTrials.gov, annotated for neurological diseases, therapeutic interventions, and control treatments. We describe our data collection process and the corpus in detail. We demonstrate its utility for NER using large language models and achieve a close-to-human performance. By bridging the gap in data resources, we hope to foster the development of text processing tools that help researchers navigate clinical trials data more easily.",
}

```
