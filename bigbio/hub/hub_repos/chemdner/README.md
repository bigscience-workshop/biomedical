
---
language: 
- en
bigbio_language: 
- English
license: unknown
multilinguality: monolingual
bigbio_license_shortname: UNKNOWN
pretty_name: CHEMDNER
homepage: https://biocreative.bioinformatics.udel.edu/resources/biocreative-iv/chemdner-corpus/
bigbio_pubmed: True
bigbio_public: True
bigbio_tasks: 
- NAMED_ENTITY_RECOGNITION
- TEXT_CLASSIFICATION
---


# Dataset Card for CHEMDNER

## Dataset Description

- **Homepage:** https://biocreative.bioinformatics.udel.edu/resources/biocreative-iv/chemdner-corpus/
- **Pubmed:** True
- **Public:** True
- **Tasks:** NER,TXTCLASS


We present the CHEMDNER corpus, a collection of 10,000 PubMed abstracts that
contain a total of 84,355 chemical entity mentions labeled manually by expert
chemistry literature curators, following annotation guidelines specifically
defined for this task. The abstracts of the CHEMDNER corpus were selected to be
representative for all major chemical disciplines. Each of the chemical entity
mentions was manually labeled according to its structure-associated chemical
entity mention (SACEM) class: abbreviation, family, formula, identifier,
multiple, systematic and trivial.



## Citation Information

```
@article{Krallinger2015,
  title        = {The CHEMDNER corpus of chemicals and drugs and its annotation principles},
  author       = {
    Krallinger, Martin and Rabal, Obdulia and Leitner, Florian and Vazquez,
    Miguel and Salgado, David and Lu, Zhiyong and Leaman, Robert and Lu, Yanan
    and Ji, Donghong and Lowe, Daniel M. and Sayle, Roger A. and
    Batista-Navarro, Riza Theresa and Rak, Rafal and Huber, Torsten and
    Rockt{"a}schel, Tim and Matos, S{'e}rgio and Campos, David and Tang,
    Buzhou and Xu, Hua and Munkhdalai, Tsendsuren and Ryu, Keun Ho and Ramanan,
    S. V. and Nathan, Senthil and {{Z}}itnik, Slavko and Bajec, Marko and
    Weber, Lutz and Irmer, Matthias and Akhondi, Saber A. and Kors, Jan A. and
    Xu, Shuo and An, Xin and Sikdar, Utpal Kumar and Ekbal, Asif and Yoshioka,
    Masaharu and Dieb, Thaer M. and Choi, Miji and Verspoor, Karin and Khabsa,
    Madian and Giles, C. Lee and Liu, Hongfang and Ravikumar, Komandur
    Elayavilli and Lamurias, Andre and Couto, Francisco M. and Dai, Hong-Jie
    and Tsai, Richard Tzong-Han and Ata, Caglar and Can, Tolga and Usi{'e},
    Anabel and Alves, Rui and Segura-Bedmar, Isabel and Mart{'i}nez, Paloma
    and Oyarzabal, Julen and Valencia, Alfonso
  },
  year         = 2015,
  month        = {Jan},
  day          = 19,
  journal      = {Journal of Cheminformatics},
  volume       = 7,
  number       = 1,
  pages        = {S2},
  doi          = {10.1186/1758-2946-7-S1-S2},
  issn         = {1758-2946},
  url          = {https://doi.org/10.1186/1758-2946-7-S1-S2},
  abstract     = {
    The automatic extraction of chemical information from text requires the
    recognition of chemical entity mentions as one of its key steps. When
    developing supervised named entity recognition (NER) systems, the
    availability of a large, manually annotated text corpus is desirable.
    Furthermore, large corpora permit the robust evaluation and comparison of
    different approaches that detect chemicals in documents. We present the
    CHEMDNER corpus, a collection of 10,000 PubMed abstracts that contain a
    total of 84,355 chemical entity mentions labeled manually by expert
    chemistry literature curators, following annotation guidelines specifically
    defined for this task. The abstracts of the CHEMDNER corpus were selected
    to be representative for all major chemical disciplines. Each of the
    chemical entity mentions was manually labeled according to its
    structure-associated chemical entity mention (SACEM) class: abbreviation,
    family, formula, identifier, multiple, systematic and trivial. The
    difficulty and consistency of tagging chemicals in text was measured using
    an agreement study between annotators, obtaining a percentage agreement of
    91. For a subset of the CHEMDNER corpus (the test set of 3,000 abstracts)
    we provide not only the Gold Standard manual annotations, but also mentions
    automatically detected by the 26 teams that participated in the BioCreative
    IV CHEMDNER chemical mention recognition task. In addition, we release the
    CHEMDNER silver standard corpus of automatically extracted mentions from
    17,000 randomly selected PubMed abstracts. A version of the CHEMDNER corpus
    in the BioC format has been generated as well. We propose a standard for
    required minimum information about entity annotations for the construction
    of domain specific corpora on chemical and drug entities. The CHEMDNER
    corpus and annotation guidelines are available at:
    ttp://www.biocreative.org/resources/biocreative-iv/chemdner-corpus/
  }
}

```
