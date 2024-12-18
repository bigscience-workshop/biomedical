---
language:
  - en
bigbio_language:
  - English
license: other
bigbio_license_shortname: other
multilinguality: monolingual
pretty_name: S800
homepage: https://species.jensenlab.org/
bigbio_pubmed: true
bigbio_public: true
bigbio_tasks:
  - NAMED_ENTITY_RECOGNITION
  - NAMED_ENTITY_DISAMBIGUATION
---


# Dataset Card for S800

## Dataset Description

- **Homepage:** https://species.jensenlab.org/
- **Pubmed:** True
- **Public:** True
- **Tasks:** NER, NED

S800 comprises 800 PubMed abstracts in which organism mentions were identified and mapped to the corresponding NCBI Taxonomy identifiers.

To increase the corpus taxonomic mention diversity the S800 abstracts were collected by selecting 100 abstracts from the following 8 categories: bacteriology, botany, entomology, medicine, mycology, protistology, virology and zoology.
S800 has been annotated with a focus at the species level; however, higher taxa mentions (such as genera, families and orders) have also been considered.


## Citation Information

```
@article{,
    title = {The SPECIES and ORGANISMS Resources for Fast and Accurate Identification of Taxonomic Names in Text},
    author = {Pafilis, Evangelos AND Frankild, Sune P. AND Fanini, Lucia AND Faulwetter, Sarah AND Pavloudi, Christina AND Vasileiadou, Aikaterini AND Arvanitidis, Christos AND Jensen, Lars Juhl},
    journal = {PLOS ONE},
    publisher = {Public Library of Science},
    year = {2013},
    month = {06},
    volume = {8},
    pages = {1-6},
    number = {6},
    url = {https://doi.org/10.1371/journal.pone.0065390},
    doi = {10.1371/journal.pone.0065390},
}
```
