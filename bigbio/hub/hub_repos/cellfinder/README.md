
---
language: 
- en
bigbio_language: 
- English
license: cc-by-sa-3.0
multilinguality: monolingual
bigbio_license_shortname: CC_BY_SA_3p0
pretty_name: CellFinder
homepage: https://www.informatik.hu-berlin.de/de/forschung/gebiete/wbi/resources/cellfinder/
bigbio_pubmed: True
bigbio_public: True
bigbio_tasks: 
- NAMED_ENTITY_RECOGNITION
---


# Dataset Card for CellFinder

## Dataset Description

- **Homepage:** https://www.informatik.hu-berlin.de/de/forschung/gebiete/wbi/resources/cellfinder/
- **Pubmed:** True
- **Public:** True
- **Tasks:** NER


The CellFinder project aims to create a stem cell data repository by linking information from existing public databases and by performing text mining on the research literature. The first version of the corpus is composed of 10 full text documents containing more than 2,100 sentences, 65,000 tokens and 5,200 annotations for entities. The corpus has been annotated with six types of entities (anatomical parts, cell components, cell lines, cell types, genes/protein and species) with an overall inter-annotator agreement around 80%.

See: https://www.informatik.hu-berlin.de/de/forschung/gebiete/wbi/resources/cellfinder/



## Citation Information

```
@inproceedings{neves2012annotating,
  title        = {Annotating and evaluating text for stem cell research},
  author       = {Neves, Mariana and Damaschun, Alexander and Kurtz, Andreas and Leser, Ulf},
  year         = 2012,
  booktitle    = {
    Proceedings of the Third Workshop on Building and Evaluation Resources for
    Biomedical Text Mining\ (BioTxtM 2012) at Language Resources and Evaluation
    (LREC). Istanbul, Turkey
  },
  pages        = {16--23},
  organization = {Citeseer}
}

```
