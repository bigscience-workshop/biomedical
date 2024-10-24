---
language:
  - en
bigbio_language:
  - English
license: cc-by-4.0
bigbio_license_shortname: CC_BY_4p0
multilinguality: monolingual
pretty_name: CoNECo
homepage: https://zenodo.org/records/11263147
bigbio_pubmed: false
bigbio_public: true
bigbio_tasks:
  - NAMED_ENTITY_RECOGNITION
  - NAMED_ENTITY_DISAMBIGUATION
paperswithcode_id: coneco
---


# Dataset Card for CoNECo

## Dataset Description

- **Homepage:** https://zenodo.org/records/11263147
- **Pubmed:** False
- **Public:** True
- **Tasks:** NER, NEN

Complex Named Entity Corpus (CoNECo) is an annotated corpus for NER and NEN of protein-containing complexes. CoNECo comprises 1,621 documents with 2,052 entities, 1,976 of which are normalized to Gene Ontology. We divided the corpus into training, development, and test sets.

## Citation Information

```
@article{10.1093/bioadv/vbae116,
    author = {Nastou, Katerina and Koutrouli, Mikaela and Pyysalo, Sampo and Jensen, Lars Juhl},
    title = "{CoNECo: A Corpus for Named Entity Recognition and Normalization of Protein Complexes}",
    journal = {Bioinformatics Advances},
    pages = {vbae116},
    year = {2024},
    month = {08},
    abstract = "{Despite significant progress in biomedical information extraction, there is a lack of resources \
for Named Entity Recognition (NER) and Normalization (NEN) of protein-containing complexes. Current resources \
inadequately address the recognition of protein-containing complex names across different organisms, underscoring \
the crucial need for a dedicated corpus.We introduce the Complex Named Entity Corpus (CoNECo), an annotated \
corpus for NER and NEN of complexes. CoNECo comprises 1,621 documents with 2,052 entities, 1,976 of which are \
normalized to Gene Ontology. We divided the corpus into training, development, and test sets and trained both a \
transformer-based and dictionary-based tagger on them. Evaluation on the test set demonstrated robust performance, \
with F-scores of 73.7\\% and 61.2\\%, respectively. Subsequently, we applied the best taggers for comprehensive \
tagging of the entire openly accessible biomedical literature.All resources, including the annotated corpus, \
training data, and code, are available to the community through Zenodo https://zenodo.org/records/11263147 and \
GitHub https://zenodo.org/records/10693653.}",
    issn = {2635-0041},
    doi = {10.1093/bioadv/vbae116},
    url = {https://doi.org/10.1093/bioadv/vbae116},
    eprint = {https://academic.oup.com/bioinformaticsadvances/advance-article-pdf/doi/10.1093/bioadv/vbae116/58869902/vbae116.pdf},
}
```
