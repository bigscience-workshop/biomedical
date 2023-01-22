
---
language: 
- en
bigbio_language: 
- English
license: cc-by-nc-2.0
multilinguality: monolingual
bigbio_license_shortname: CC_BY_NC_2p0
pretty_name: SciFact
homepage: https://scifact.apps.allenai.org/
bigbio_pubmed: False
bigbio_public: True
bigbio_tasks: 
- TEXT_PAIRS_CLASSIFICATION
---


# Dataset Card for SciFact

## Dataset Description

- **Homepage:** https://scifact.apps.allenai.org/
- **Pubmed:** False
- **Public:** True
- **Tasks:** TXT2CLASS


### Scifact Corpus Source

        SciFact is a dataset of 1.4K expert-written scientific claims paired with evidence-containing abstracts, and annotated with labels and rationales.
     This config has abstracts and document ids.
    

### Scifact Claims Source

    {_DESCRIPTION_BASE} This config connects the claims to the evidence and doc ids.
    

### Scifact Rationale Bigbio Pairs

    {_DESCRIPTION_BASE} This task is the following: given a claim and a text span composed of one or more sentences from an abstract, predict a label from ("rationale", "not_rationale") indicating if the span is evidence (can be supporting or refuting) for the claim. This roughly corresponds to the second task outlined in Section 5 of the paper."
    

### Scifact Labelprediction Bigbio Pairs

    {_DESCRIPTION_BASE} This task is the following: given a claim and a text span composed of one or more sentences from an abstract, predict a label from ("SUPPORT", "NOINFO", "CONTRADICT") indicating if the span supports, provides no info, or contradicts the claim. This roughly corresponds to the thrid task outlined in Section 5 of the paper.
    




## Citation Information

```
@article{wadden2020fact,
  author    = {David Wadden and Shanchuan Lin and Kyle Lo and Lucy Lu Wang and Madeleine van Zuylen and Arman Cohan and Hannaneh Hajishirzi},
  title     = {Fact or Fiction: Verifying Scientific Claims},
  year      = {2020},
  address   = {Online},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2020.emnlp-main.609},
  doi       = {10.18653/v1/2020.emnlp-main.609},
  pages     = {7534--7550},
  biburl    = {},
  bibsource = {}
}

```
