
---
language: 
- en
bigbio_language: 
- English
license: unknown
multilinguality: monolingual
bigbio_license_shortname: UNKNOWN
pretty_name: LLL05
homepage: http://genome.jouy.inra.fr/texte/LLLchallenge
bigbio_pubmed: True
bigbio_public: True
bigbio_tasks: 
- RELATION_EXTRACTION
---


# Dataset Card for LLL05

## Dataset Description

- **Homepage:** http://genome.jouy.inra.fr/texte/LLLchallenge
- **Pubmed:** True
- **Public:** True
- **Tasks:** RE


The LLL05 challenge task is to learn rules to extract protein/gene interactions from biology abstracts from the Medline
bibliography database. The goal of the challenge is to test the ability of the participating IE systems to identify the
interactions and the gene/proteins that interact. The participants will test their IE patterns on a test set with the
aim of extracting the correct agent and target.The challenge focuses on information extraction of gene interactions in
Bacillus subtilis. Extracting gene interaction is the most popular event IE task in biology. Bacillus subtilis (Bs) is
a model bacterium and many papers have been published on direct gene interactions involved in sporulation. The gene
interactions are generally mentioned in the abstract and the full text of the paper is not needed. Extracting gene
interaction means, extracting the agent (proteins) and the target (genes) of all couples of genic interactions from
sentences.



## Citation Information

```
    @article{article,
    author = {Nédellec, C.},
    year = {2005},
    month = {01},
    pages = {},
    title = {Learning Language in Logic - Genic Interaction Extraction Challenge},
    journal = {Proceedings of the Learning Language in Logic 2005 Workshop at the         International Conference on Machine Learning}
}

```
