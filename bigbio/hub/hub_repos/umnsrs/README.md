
---
language: 
- en
bigbio_language: 
- English
license: cc0-1.0
multilinguality: monolingual
bigbio_license_shortname: CC0_1p0
pretty_name: UMNSRS
homepage: https://conservancy.umn.edu/handle/11299/196265/
bigbio_pubmed: False
bigbio_public: True
bigbio_tasks: 
- SEMANTIC_SIMILARITY
---


# Dataset Card for UMNSRS

## Dataset Description

- **Homepage:** https://conservancy.umn.edu/handle/11299/196265/
- **Pubmed:** False
- **Public:** True
- **Tasks:** STS


UMNSRS, developed by Pakhomov, et al., consists of 725 clinical term pairs whose semantic similarity and relatedness.
The similarity and relatedness of each term pair was annotated based on a continuous scale by having the resident touch
a bar on a touch sensitive computer screen to indicate the degree of similarity or relatedness.
The following subsets are available:
- similarity: A set of 566 UMLS concept pairs manually rated for semantic similarity (e.g. whale-dolphin) using a
  continuous response scale.
- relatedness: A set of 588 UMLS concept pairs manually rated for semantic relatedness (e.g. needle-thread) using a
  continuous response scale.
- similarity_mod: Modification of the UMNSRS-Similarity dataset to exclude control samples and those pairs that did not
  match text in clinical, biomedical and general English corpora. Exact modifications are detailed in the paper (Corpus
  Domain Effects on Distributional Semantic Modeling of Medical Terms. Serguei V.S. Pakhomov, Greg Finley, Reed McEwan,
  Yan Wang, and Genevieve B. Melton. Bioinformatics. 2016; 32(23):3635-3644). The resulting dataset contains 449 pairs.
- relatedness_mod: Modification of the UMNSRS-Relatedness dataset to exclude control samples and those pairs that did
  not match text in clinical, biomedical and general English corpora. Exact modifications are detailed in the paper
  (Corpus Domain Effects on Distributional Semantic Modeling of Medical Terms. Serguei V.S. Pakhomov, Greg Finley,
  Reed McEwan, Yan Wang, and Genevieve B. Melton. Bioinformatics. 2016; 32(23):3635-3644).
  The resulting dataset contains 458 pairs.



## Citation Information

```
@inproceedings{pakhomov2010semantic,
  title={Semantic similarity and relatedness between clinical terms: an experimental study},
  author={Pakhomov, Serguei and McInnes, Bridget and Adam, Terrence and Liu, Ying and Pedersen, Ted and Melton,   Genevieve B},
  booktitle={AMIA annual symposium proceedings},
  volume={2010},
  pages={572},
  year={2010},
  organization={American Medical Informatics Association}
}

```
