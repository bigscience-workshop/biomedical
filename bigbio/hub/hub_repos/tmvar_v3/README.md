
---
language: 
- en
bigbio_language: 
- English
license: unknown
multilinguality: monolingual
bigbio_license_shortname: UNKNOWN
pretty_name: tmVar v3
homepage: https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/tmvar/
bigbio_pubmed: True
bigbio_public: True
bigbio_tasks: 
- NAMED_ENTITY_RECOGNITION
- NAMED_ENTITY_DISAMBIGUATION
---


# Dataset Card for tmVar v3

## Dataset Description

- **Homepage:** https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/tmvar/
- **Pubmed:** True
- **Public:** True
- **Tasks:** NER,NED


This dataset contains 500 PubMed articles manually annotated with mutation mentions of various kinds and dbsnp normalizations for each of them.  In addition, it contains variant normalization options such as allele-specific identifiers from the ClinGen Allele Registry It can be used for NER tasks and NED tasks, This dataset does NOT have splits.



## Citation Information

```
@misc{https://doi.org/10.48550/arxiv.2204.03637,
  title        = {tmVar 3.0: an improved variant concept recognition and normalization tool},
  author       = {
    Wei, Chih-Hsuan and Allot, Alexis and Riehle, Kevin and Milosavljevic,
    Aleksandar and Lu, Zhiyong
  },
  year         = 2022,
  publisher    = {arXiv},
  doi          = {10.48550/ARXIV.2204.03637},
  url          = {https://arxiv.org/abs/2204.03637},
  copyright    = {Creative Commons Attribution 4.0 International},
  keywords     = {
    Computation and Language (cs.CL), FOS: Computer and information sciences,
    FOS: Computer and information sciences
  }
}


```
