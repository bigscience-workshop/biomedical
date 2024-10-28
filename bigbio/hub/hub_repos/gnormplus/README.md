
---
language: 
- en
bigbio_language: 
- English
license: unknown
multilinguality: monolingual
bigbio_license_shortname: UNKNOWN
pretty_name: GNormPlus
homepage: https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/gnormplus/
bigbio_pubmed: True
bigbio_public: True
bigbio_tasks: 
- NAMED_ENTITY_RECOGNITION
- NAMED_ENTITY_DISAMBIGUATION
---


# Dataset Card for GNormPlus

## Dataset Description

- **Homepage:** https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/gnormplus/
- **Pubmed:** True
- **Public:** True
- **Tasks:** NER,NED


We re-annotated two existing gene corpora. The BioCreative II GN corpus is a widely used data set for benchmarking GN
tools and includes document-level annotations for a total of 543 articles (281 in its training set; and 262 in test).
The Citation GIA Test Collection was recently created for gene indexing at the NLM and includes 151 PubMed abstracts
with both mention-level and document-level annotations. They are selected because both have a focus on human genes.
For both corpora, we added annotations of gene families and protein domains. For the BioCreative GN corpus, we also
added mention-level gene annotations. As a result, in our new corpus, there are a total of 694 PubMed articles.
PubTator was used as our annotation tool along with BioC formats.



## Citation Information

```
@Article{Wei2015,
author={Wei, Chih-Hsuan and Kao, Hung-Yu and Lu, Zhiyong},
title={GNormPlus: An Integrative Approach for Tagging Genes, Gene Families, and Protein Domains},
journal={BioMed Research International},
year={2015},
month={Aug},
day={25},
publisher={Hindawi Publishing Corporation},
volume={2015},
pages={918710},
issn={2314-6133},
doi={10.1155/2015/918710},
url={https://doi.org/10.1155/2015/918710}
}

```
