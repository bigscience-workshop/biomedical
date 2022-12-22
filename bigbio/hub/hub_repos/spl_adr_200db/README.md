
---
language: 
- en
bigbio_language: 
- English
license: cc0-1.0
multilinguality: monolingual
bigbio_license_shortname: CC0_1p0
pretty_name: SPL ADR
homepage: https://bionlp.nlm.nih.gov/tac2017adversereactions/
bigbio_pubmed: False
bigbio_public: True
bigbio_tasks: 
- NAMED_ENTITY_RECOGNITION
- NAMED_ENTITY_DISAMBIGUATION
- RELATION_EXTRACTION
---


# Dataset Card for SPL ADR

## Dataset Description

- **Homepage:** https://bionlp.nlm.nih.gov/tac2017adversereactions/
- **Pubmed:** False
- **Public:** True
- **Tasks:** NER,NED,RE


The United States Food and Drug Administration (FDA) partnered with the National Library
of Medicine to create a pilot dataset containing standardised information about known
adverse reactions for 200 FDA-approved drugs. The Structured Product Labels (SPLs),
the documents FDA uses to exchange information about drugs and other products, were
manually annotated for adverse reactions at the mention level to facilitate development
and evaluation of text mining tools for extraction of ADRs from all SPLs.  The ADRs were
then normalised to the Unified Medical Language System (UMLS) and to the Medical
Dictionary for Regulatory Activities (MedDRA).



## Citation Information

```
@article{demner2018dataset,
  author    = {Demner-Fushman, Dina and Shooshan, Sonya and Rodriguez, Laritza and Aronson,
               Alan and Lang, Francois and Rogers, Willie and Roberts, Kirk and Tonning, Joseph},
  title     = {A dataset of 200 structured product labels annotated for adverse drug reactions},
  journal   = {Scientific Data},
  volume    = {5},
  year      = {2018},
  month     = {01},
  pages     = {180001},
  url       = {
    https://www.researchgate.net/publication/322810855_A_dataset_of_200_structured_product_labels_annotated_for_adverse_drug_reactions
  },
  doi       = {10.1038/sdata.2018.1}
}

```
