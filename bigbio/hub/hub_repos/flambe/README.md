---
language:
  - en 
bigbio_language:
  - English
license: cc-by-4.0
bigbio_license_shortname: APACHE_2p0
multilinguality: monolingual
pretty_name: FlaMBe
homepage: https://github.com/ylaboratory/flambe
bigbio_pubmed: false
bigbio_public: true
bigbio_tasks:
  - NAMED_ENTITY_RECOGNITION
  - NAMED_ENTITY_DISAMBIGUATION 
---


# Dataset Card for SciTail

## Dataset Description

- **Homepage:** https://github.com/ylaboratory/flambe
- **Pubmed:** False
- **Public:** True
- **Tasks:** TE [This needs to be a comma delimitted string of task short names]


FlaMBe is a dataset aimed at procedural knowledge extraction from biomedical texts, particularly focusing on single cell research methodologies described in academic papers. It includes annotations from 55 full-text articles and 1,195 abstracts, covering nearly 710,000 tokens, and is  distinguished by its comprehensive named entity recognition (NER) and disambiguation (NED) for  tissue/cell types, software tools, and computational methods. This dataset, to our knowledge, is the largest of its kind for tissue/cell types, links entities to identifiers in relevant knowledge  bases and annotates nearly 400 workflow relations between tool-context pairs. 


## Citation Information

```
@inproceedings{,
  author    = {Dannenfelser, Ruth and Zhong, Jeffrey and Zhang, Ran and Yao, Vicky},
  title     = {Into the Single Cell Multiverse: an End-to-End Dataset for Procedural Knowledge Extraction in Biomedical Texts},
  publisher   = {Advances in Neural Information Processing Systems},
  volume    = {36},
  year      = {2024},
  url       = {https://proceedings.neurips.cc/paper_files/paper/2023/file/23e3d86c9a19d0caf2ec997e73dfcfbd-Paper-Datasets_and_Benchmarks.pdf},
}
```
