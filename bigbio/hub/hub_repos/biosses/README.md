---
language: 
- en
bigbio_language:
- English
license: gpl-3.0
multilinguality: monolingual
bigbio_license_shortname: GPL_3p0
pretty_name: BIOSSES
homepage: https://tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.html
bigbio_pubmed: false
bigbio_public: true
bigbio_tasks:
- SEMANTIC_SIMILARITY
---


# Dataset Card for BIOSSES


## Dataset Description

- **Homepage:** https://tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.html
- **Pubmed:** True
- **Public:** True
- **Tasks:** STS

BIOSSES computes similarity of biomedical sentences by utilizing WordNet as the general domain ontology and UMLS as the biomedical domain specific ontology. The original paper outlines the approaches with respect to using annotator score as golden standard. Source view will return all annotator score individually whereas the Bigbio view will return the mean of the annotator score.


## Citation Information

```
@article{souganciouglu2017biosses,
  title={BIOSSES: a semantic sentence similarity estimation system for the biomedical domain},
  author={Soğancıoğlu, Gizem, Hakime Öztürk, and Arzucan Özgür},
  journal={Bioinformatics},
  volume={33},
  number={14},
  pages={i49--i58},
  year={2017},
  publisher={Oxford University Press}
}
```
