
---
language: 
- en
bigbio_language: 
- English
license: other
multilinguality: monolingual
bigbio_license_shortname: GENIA_PROJECT_LICENSE
pretty_name: GENIA Relation Corpus
homepage: http://www.geniaproject.org/genia-corpus/relation-corpus
bigbio_pubmed: True
bigbio_public: True
bigbio_tasks: 
- RELATION_EXTRACTION
---


# Dataset Card for GENIA Relation Corpus

## Dataset Description

- **Homepage:** http://www.geniaproject.org/genia-corpus/relation-corpus
- **Pubmed:** True
- **Public:** True
- **Tasks:** RE


The extraction of various relations stated to hold between biomolecular entities is one of the most frequently
addressed information extraction tasks in domain studies. Typical relation extraction targets involve protein-protein
interactions or gene regulatory relations. However, in the GENIA corpus, such associations involving change in the
state or properties of biomolecules are captured in the event annotation.

The GENIA corpus relation annotation aims to complement the event annotation of the corpus by capturing (primarily)
static relations, relations such as part-of that hold between entities without (necessarily) involving change.



## Citation Information

```
@inproceedings{pyysalo-etal-2009-static,
    title = "Static Relations: a Piece in the Biomedical Information Extraction Puzzle",
    author = "Pyysalo, Sampo  and
      Ohta, Tomoko  and
      Kim, Jin-Dong  and
      Tsujii, Jun{'}ichi",
    booktitle = "Proceedings of the {B}io{NLP} 2009 Workshop",
    month = jun,
    year = "2009",
    address = "Boulder, Colorado",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W09-1301",
    pages = "1--9",
}

@article{article,
author = {Ohta, Tomoko and Pyysalo, Sampo and Kim, Jin-Dong and Tsujii, Jun'ichi},
year = {2010},
month = {10},
pages = {917-28},
title = {A reevaluation of biomedical named entity - term relations},
volume = {8},
journal = {Journal of bioinformatics and computational biology},
doi = {10.1142/S0219720010005014}
}

@MISC{Hoehndorf_applyingontology,
    author = {Robert Hoehndorf and Axel-cyrille Ngonga Ngomo and Sampo Pyysalo and Tomoko Ohta and Anika Oellrich and
    Dietrich Rebholz-schuhmann},
    title = {Applying ontology design patterns to the implementation of relations in GENIA},
    year = {}
}

```
