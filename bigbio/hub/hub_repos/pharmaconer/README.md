
---
language: 
- es
bigbio_language: 
- Spanish
license: cc-by-4.0
multilinguality: monolingual
bigbio_license_shortname: CC_BY_4p0
pretty_name: PharmaCoNER
homepage: https://temu.bsc.es/pharmaconer/index.php/datasets/
bigbio_pubmed: False
bigbio_public: True
bigbio_tasks: 
- NAMED_ENTITY_RECOGNITION
- TEXT_CLASSIFICATION
---


# Dataset Card for PharmaCoNER

## Dataset Description

- **Homepage:** https://temu.bsc.es/pharmaconer/index.php/datasets/
- **Pubmed:** False
- **Public:** True
- **Tasks:** NER,TXTCLASS


### Subtrack 1

PharmaCoNER: Pharmacological Substances, Compounds and Proteins Named Entity Recognition track

This dataset is designed for the PharmaCoNER task, sponsored by Plan de Impulso de las Tecnologías del Lenguaje.

It is a manually classified collection of clinical case studies derived from the Spanish Clinical Case Corpus (SPACCC), an open access electronic library that gathers Spanish medical publications from SciELO (Scientific Electronic Library Online).

The annotation of the entire set of entity mentions was carried out by medicinal chemistry experts and it includes the following 4 entity types: NORMALIZABLES, NO_NORMALIZABLES, PROTEINAS and UNCLEAR.

The PharmaCoNER corpus contains a total of 396,988 words and 1,000 clinical cases that have been randomly sampled into 3 subsets. The training set contains 500 clinical cases, while the development and test sets contain 250 clinical cases each.

For further information, please visit https://temu.bsc.es/pharmaconer/ or send an email to encargo-pln-life@bsc.es


SUBTRACK 1: NER offset and entity type classification

The first subtrack consists in the classical entity-based or instanced-based evaluation that requires that system outputs match exactly the beginning and end locations of each entity tag, as well as match the entity annotation type of the gold standard annotations.


### Subtrack 2

PharmaCoNER: Pharmacological Substances, Compounds and Proteins Named Entity Recognition track

This dataset is designed for the PharmaCoNER task, sponsored by Plan de Impulso de las Tecnologías del Lenguaje.

It is a manually classified collection of clinical case studies derived from the Spanish Clinical Case Corpus (SPACCC), an open access electronic library that gathers Spanish medical publications from SciELO (Scientific Electronic Library Online).

The annotation of the entire set of entity mentions was carried out by medicinal chemistry experts and it includes the following 4 entity types: NORMALIZABLES, NO_NORMALIZABLES, PROTEINAS and UNCLEAR.

The PharmaCoNER corpus contains a total of 396,988 words and 1,000 clinical cases that have been randomly sampled into 3 subsets. The training set contains 500 clinical cases, while the development and test sets contain 250 clinical cases each.

For further information, please visit https://temu.bsc.es/pharmaconer/ or send an email to encargo-pln-life@bsc.es


SUBTRACK 2: CONCEPT INDEXING

In the second subtask, a list of unique SNOMED concept identifiers have to be generated for each document. The predictions are compared to the manually annotated concept ids corresponding to chemical compounds and pharmacological substances.


### Full Task

PharmaCoNER: Pharmacological Substances, Compounds and Proteins Named Entity Recognition track

This dataset is designed for the PharmaCoNER task, sponsored by Plan de Impulso de las Tecnologías del Lenguaje.

It is a manually classified collection of clinical case studies derived from the Spanish Clinical Case Corpus (SPACCC), an open access electronic library that gathers Spanish medical publications from SciELO (Scientific Electronic Library Online).

The annotation of the entire set of entity mentions was carried out by medicinal chemistry experts and it includes the following 4 entity types: NORMALIZABLES, NO_NORMALIZABLES, PROTEINAS and UNCLEAR.

The PharmaCoNER corpus contains a total of 396,988 words and 1,000 clinical cases that have been randomly sampled into 3 subsets. The training set contains 500 clinical cases, while the development and test sets contain 250 clinical cases each.

For further information, please visit https://temu.bsc.es/pharmaconer/ or send an email to encargo-pln-life@bsc.es


SUBTRACK 1: NER offset and entity type classification

The first subtrack consists in the classical entity-based or instanced-based evaluation that requires that system outputs match exactly the beginning and end locations of each entity tag, as well as match the entity annotation type of the gold standard annotations.


SUBTRACK 2: CONCEPT INDEXING

In the second subtask, a list of unique SNOMED concept identifiers have to be generated for each document. The predictions are compared to the manually annotated concept ids corresponding to chemical compounds and pharmacological substances.





## Citation Information

```
@inproceedings{gonzalez2019pharmaconer,
    title = "PharmaCoNER: Pharmacological Substances, Compounds and proteins Named Entity Recognition track",
    author = "Gonzalez-Agirre, Aitor  and
      Marimon, Montserrat  and
      Intxaurrondo, Ander  and
      Rabal, Obdulia  and
      Villegas, Marta  and
      Krallinger, Martin",
    booktitle = "Proceedings of The 5th Workshop on BioNLP Open Shared Tasks",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D19-5701",
    doi = "10.18653/v1/D19-5701",
    pages = "1--10",
}

```
