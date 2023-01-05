
---
language: 
- en
- de
bigbio_language: 
- English
- German
license: unknown
multilinguality: multilingual
bigbio_license_shortname: UNKNOWN
pretty_name: MuchMore
homepage: https://muchmore.dfki.de/resources1.htm
bigbio_pubmed: True
bigbio_public: True
bigbio_tasks: 
- TRANSLATION
- NAMED_ENTITY_RECOGNITION
- NAMED_ENTITY_DISAMBIGUATION
- RELATION_EXTRACTION
---


# Dataset Card for MuchMore

## Dataset Description

- **Homepage:** https://muchmore.dfki.de/resources1.htm
- **Pubmed:** True
- **Public:** True
- **Tasks:** TRANSL,NER,NED,RE


The corpus used in the MuchMore project is a parallel corpus of English-German scientific
medical abstracts obtained from the Springer Link web site. The corpus consists
approximately of 1 million tokens for each language. Abstracts are from 41 medical
journals, each of which constitutes a relatively homogeneous medical sub-domain (e.g.
Neurology, Radiology, etc.). The corpus of downloaded HTML documents is normalized in
various ways, in order to produce a clean, plain text version, consisting of a title, abstract
and keywords. Additionally, the corpus was aligned on the sentence level.

Automatic (!) annotation includes: Part-of-Speech; Morphology (inflection and
decomposition); Chunks; Semantic Classes (UMLS: Unified Medical Language System,
MeSH: Medical Subject Headings, EuroWordNet); Semantic Relations from UMLS.



## Citation Information

```
@inproceedings{buitelaar2003multi,
  title={A multi-layered, xml-based approach to the integration of linguistic and semantic annotations},
  author={Buitelaar, Paul and Declerck, Thierry and Sacaleanu, Bogdan and Vintar, {{S}}pela and Raileanu, Diana and Crispi, Claudia},
  booktitle={Proceedings of EACL 2003 Workshop on Language Technology and the Semantic Web (NLPXML'03), Budapest, Hungary},
  year={2003}
}

```
