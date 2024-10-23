
---
language: 
- en
bigbio_language: 
- English
license: cc0-1.0
multilinguality: monolingual
bigbio_license_shortname: CC0_1p0
pretty_name: MedMentions
homepage: https://github.com/chanzuckerberg/MedMentions
bigbio_pubmed: True
bigbio_public: True
bigbio_tasks: 
- NAMED_ENTITY_DISAMBIGUATION
- NAMED_ENTITY_RECOGNITION
---


# Dataset Card for MedMentions

## Dataset Description

- **Homepage:** https://github.com/chanzuckerberg/MedMentions
- **Pubmed:** True
- **Public:** True
- **Tasks:** NED,NER


MedMentions is a new manually annotated resource for the recognition of biomedical concepts.
What distinguishes MedMentions from other annotated biomedical corpora is its size (over 4,000
abstracts and over 350,000 linked mentions), as well as the size of the concept ontology (over
3 million concepts from UMLS 2017) and its broad coverage of biomedical disciplines.

Corpus: The MedMentions corpus consists of 4,392 papers (Titles and Abstracts) randomly selected
from among papers released on PubMed in 2016, that were in the biomedical field, published in
the English language, and had both a Title and an Abstract.

Annotators: We recruited a team of professional annotators with rich experience in biomedical
content curation to exhaustively annotate all UMLS® (2017AA full version) entity mentions in
these papers.

Annotation quality: We did not collect stringent IAA (Inter-annotator agreement) data. To gain
insight on the annotation quality of MedMentions, we randomly selected eight papers from the
annotated corpus, containing a total of 469 concepts. Two biologists ('Reviewer') who did not
participate in the annotation task then each reviewed four papers. The agreement between
Reviewers and Annotators, an estimate of the Precision of the annotations, was 97.3%.



## Citation Information

```
@misc{mohan2019medmentions,
      title={MedMentions: A Large Biomedical Corpus Annotated with UMLS Concepts},
      author={Sunil Mohan and Donghui Li},
      year={2019},
      eprint={1902.09476},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

```
