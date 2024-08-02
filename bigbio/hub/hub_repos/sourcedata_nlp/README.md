---
language:
  - en 
bigbio_language:
  - English
license: "CC-BY 4.0"
bigbio_license_shortname: cc-by-4.0
multilinguality: monolingual
pretty_name: SourceData NLP
homepage: https://sourcedata.embo.org/
bigbio_pubmed: false
bigbio_public: true
bigbio_tasks:
  - NAMED_ENTITY_RECOGNITION
  - NAMED_ENTITY_DISAMBIGUATION
paperswithcode_id: sourcedata-nlp
---


# Dataset Card for SourceData NLP

## Dataset Description

- **Homepage:** https://sourcedata.embo.org/
- **Pubmed:** False
- **Public:** True
- **Tasks:** NER,NED


SourceData-NLP is a named entity recognition and entity linking/disambiguation dataset produced through the routine curation of papers during the publication process. All annotations are in figure legends from published papers in molecular and cell biologyThe dataset consists of eight classes of biomedical entities (small molecules, gene products, subcellular components, cell lines, cell types, tissues, organisms, and diseases), their role in the experimental design, and the nature of the experimental method as an additional class. SourceData-NLP contains more than 620,000 annotated biomedical entities, curated from 18,689 figures in 3,223 papers in molecular and cell biology.


## Citation Information

```
@article{abreu2023sourcedata,
  title={The SourceData-NLP dataset: integrating curation into scientific publishing for training large language models},
  author={Abreu-Vicente, Jorge and Sonntag, Hannah and Eidens, Thomas and Lemberger, Thomas},
  journal={arXiv preprint arXiv:2310.20440},
  year={2023}
}
```