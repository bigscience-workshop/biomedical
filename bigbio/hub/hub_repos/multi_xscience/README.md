
---
language: 
- en
bigbio_language: 
- English
license: mit
multilinguality: monolingual
bigbio_license_shortname: MIT
pretty_name: Multi-XScience
homepage: https://github.com/yaolu/Multi-XScience
bigbio_pubmed: False
bigbio_public: True
bigbio_tasks: 
- PARAPHRASING
- SUMMARIZATION
---


# Dataset Card for Multi-XScience

## Dataset Description

- **Homepage:** https://github.com/yaolu/Multi-XScience
- **Pubmed:** False
- **Public:** True
- **Tasks:** PARA,SUM


Multi-document summarization is a challenging task for which there exists little large-scale datasets. 
We propose Multi-XScience, a large-scale multi-document summarization dataset created from scientific articles. 
Multi-XScience introduces a challenging multi-document summarization task: writing the related-work section 
of a paper based on its abstract and the articles it references. Our work is inspired by extreme summarization, 
a dataset construction protocol that favours abstractive modeling approaches. Descriptive statistics and 
empirical results---using several state-of-the-art models trained on the Multi-XScience dataset---reveal t
hat Multi-XScience is well suited for abstractive models.



## Citation Information

```
@misc{https://doi.org/10.48550/arxiv.2010.14235,
  doi = {10.48550/ARXIV.2010.14235},
  
  url = {https://arxiv.org/abs/2010.14235},
  
  author = {Lu, Yao and Dong, Yue and Charlin, Laurent},
  
  keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Multi-XScience: A Large-scale Dataset for Extreme Multi-document Summarization of Scientific Articles},
  
  publisher = {arXiv},
  
  year = {2020},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```
