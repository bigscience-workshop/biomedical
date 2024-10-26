
---
language: 
- en
bigbio_language: 
- English
license: unknown
multilinguality: monolingual
bigbio_license_shortname: UNKNOWN
pretty_name: HPRD50
homepage: 
bigbio_pubmed: True
bigbio_public: True
bigbio_tasks: 
- RELATION_EXTRACTION
- NAMED_ENTITY_RECOGNITION
---


# Dataset Card for HPRD50

## Dataset Description

- **Homepage:** 
- **Pubmed:** True
- **Public:** True
- **Tasks:** RE,NER


HPRD50 is a dataset of randomly selected, hand-annotated abstracts of biomedical papers
referenced by the Human Protein Reference Database (HPRD). It is parsed in XML format,
splitting each abstract into sentences, and in each sentence there may be entities and
interactions between those entities. In this particular dataset, entities are all
proteins and interactions are thus protein-protein interactions.

Moreover, all entities are normalized to the HPRD database. These normalized terms are
stored in each entity's 'type' attribute in the source XML. This means the dataset can
determine e.g. that "Janus kinase 2" and "Jak2" are referencing the same normalized
entity.

Because the dataset contains entities and relations, it is suitable for Named Entity
Recognition and Relation Extraction.



## Citation Information

```
@article{fundel2007relex,
  title={RelEx—Relation extraction using dependency parse trees},
  author={Fundel, Katrin and K{"u}ffner, Robert and Zimmer, Ralf},
  journal={Bioinformatics},
  volume={23},
  number={3},
  pages={365--371},
  year={2007},
  publisher={Oxford University Press}
}

```
