
---
language: 
- en
bigbio_language: 
- English
license: other
multilinguality: monolingual
bigbio_license_shortname: UMLS_LICENSE
pretty_name: MSH WSD
homepage: https://lhncbc.nlm.nih.gov/ii/areas/WSD/collaboration.html
bigbio_pubmed: True
bigbio_public: False
bigbio_tasks: 
- NAMED_ENTITY_DISAMBIGUATION
---


# Dataset Card for MSH WSD

## Dataset Description

- **Homepage:** https://lhncbc.nlm.nih.gov/ii/areas/WSD/collaboration.html
- **Pubmed:** True
- **Public:** False
- **Tasks:** NED


Evaluation of Word Sense Disambiguation methods (WSD) in the biomedical domain is difficult because the available
resources are either too small or too focused on specific types of entities (e.g. diseases or genes). We have
developed a method that can be used to automatically develop a WSD test collection using the Unified Medical Language
System (UMLS) Metathesaurus and the manual MeSH indexing of MEDLINE. The resulting dataset is called MSH WSD and
consists of 106 ambiguous abbreviations, 88 ambiguous terms and 9 which are a combination of both, for a total of 203
ambiguous words. Each instance containing the ambiguous word was assigned a CUI from the 2009AB version of the UMLS.
For each ambiguous term/abbreviation, the data set contains a maximum of 100 instances per sense obtained from
MEDLINE; totaling 37,888 ambiguity cases in 37,090 MEDLINE citations.



## Citation Information

```
@article{jimeno2011exploiting,
  title={Exploiting MeSH indexing in MEDLINE to generate a data set for word sense disambiguation},
  author={Jimeno-Yepes, Antonio J and McInnes, Bridget T and Aronson, Alan R},
  journal={BMC bioinformatics},
  volume={12},
  number={1},
  pages={1--14},
  year={2011},
  publisher={BioMed Central}
}

```
