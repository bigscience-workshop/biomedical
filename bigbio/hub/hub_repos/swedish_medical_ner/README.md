
---
language: 
- sv
bigbio_language: 
- Swedish
license: cc-by-sa-4.0
multilinguality: monolingual
bigbio_license_shortname: CC_BY_SA_4p0
pretty_name: Swedish Medical NER
homepage: https://github.com/olofmogren/biomedical-ner-data-swedish/
bigbio_pubmed: False
bigbio_public: True
bigbio_tasks: 
- NAMED_ENTITY_RECOGNITION
---


# Dataset Card for Swedish Medical NER

## Dataset Description

- **Homepage:** https://github.com/olofmogren/biomedical-ner-data-swedish/
- **Pubmed:** False
- **Public:** True
- **Tasks:** NER


swedish_medical_ner is Named Entity Recognition dataset on medical text in Swedish. 
It consists three subsets which are in turn derived from three different sources 
respectively: the Swedish Wikipedia (a.k.a. wiki), Läkartidningen (a.k.a. lt), 
and 1177 Vårdguiden (a.k.a. 1177). While the Swedish Wikipedia and Läkartidningen 
subsets in total contains over 790000 sequences with 60 characters each, 
the 1177 Vårdguiden subset is manually annotated and contains 927 sentences, 
2740 annotations, out of which 1574 are disorder and findings, 546 are 
pharmaceutical drug, and 620 are body structure.

Texts from both Swedish Wikipedia and Läkartidningen were automatically annotated 
using a list of medical seed terms. Sentences from 1177 Vårdguiden were manuually 
annotated.



## Citation Information

```
@inproceedings{almgren-etal-2016-named,
    author = {
        Almgren, Simon and
        Pavlov, Sean and
        Mogren, Olof
    },
    title     = {Named Entity Recognition in Swedish Medical Journals with Deep Bidirectional Character-Based LSTMs},
    booktitle = {Proceedings of the Fifth Workshop on Building and Evaluating Resources for Biomedical Text Mining (BioTxtM 2016)},
    publisher = {The COLING 2016 Organizing Committee},
    pages     = {30-39},
    year      = {2016},
    month     = {12},
    url       = {https://aclanthology.org/W16-5104},
    eprint    = {https://aclanthology.org/W16-5104.pdf}
}

```
