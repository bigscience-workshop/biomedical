
---
language: 
- en
bigbio_language: 
- English
license: gpl-3.0
multilinguality: monolingual
bigbio_license_shortname: GPL_3p0
pretty_name: Hallmarks of Cancer
homepage: https://github.com/sb895/Hallmarks-of-Cancer
bigbio_pubmed: True
bigbio_public: True
bigbio_tasks: 
- TEXT_CLASSIFICATION
---


# Dataset Card for Hallmarks of Cancer

## Dataset Description

- **Homepage:** https://github.com/sb895/Hallmarks-of-Cancer
- **Pubmed:** True
- **Public:** True
- **Tasks:** TXTCLASS


The Hallmarks of Cancer (HOC) Corpus consists of 1852 PubMed publication
abstracts manually annotated by experts according to a taxonomy. The taxonomy
consists of 37 classes in a hierarchy. Zero or more class labels are assigned
to each sentence in the corpus. The labels are found under the "labels"
directory, while the tokenized text can be found under "text" directory.
The filenames are the corresponding PubMed IDs (PMID).



## Citation Information

```
@article{DBLP:journals/bioinformatics/BakerSGAHSK16,
  author    = {Simon Baker and
               Ilona Silins and
               Yufan Guo and
               Imran Ali and
               Johan H{"{o}}gberg and
               Ulla Stenius and
               Anna Korhonen},
  title     = {Automatic semantic classification of scientific literature
               according to the hallmarks of cancer},
  journal   = {Bioinform.},
  volume    = {32},
  number    = {3},
  pages     = {432--440},
  year      = {2016},
  url       = {https://doi.org/10.1093/bioinformatics/btv585},
  doi       = {10.1093/bioinformatics/btv585},
  timestamp = {Thu, 14 Oct 2021 08:57:44 +0200},
  biburl    = {https://dblp.org/rec/journals/bioinformatics/BakerSGAHSK16.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

```
