
---
language: 
- en
bigbio_language: 
- English
license: other
multilinguality: monolingual
bigbio_license_shortname: UMLS_LICENSE
pretty_name: NLM WSD
homepage: https://lhncbc.nlm.nih.gov/restricted/ii/areas/WSD/index.html
bigbio_pubmed: True
bigbio_public: False
bigbio_tasks: 
- NAMED_ENTITY_DISAMBIGUATION
---


# Dataset Card for NLM WSD

## Dataset Description

- **Homepage:** https://lhncbc.nlm.nih.gov/restricted/ii/areas/WSD/index.html
- **Pubmed:** True
- **Public:** False
- **Tasks:** NED


In order to support research investigating the automatic resolution of word sense ambiguity using natural language
processing techniques, we have constructed this test collection of medical text in which the ambiguities were resolved
by hand. Evaluators were asked to examine instances of an ambiguous word and determine the sense intended by selecting
the Metathesaurus concept (if any) that best represents the meaning of that sense. The test collection consists of 50
highly frequent ambiguous UMLS concepts from 1998 MEDLINE. Each of the 50 ambiguous cases has 100 ambiguous instances
randomly selected from the 1998 MEDLINE citations. For a total of 5,000 instances. We had a total of 11 evaluators of
which 8 completed 100% of the 5,000 instances, 1 completed 56%, 1 completed 44%, and the final evaluator completed 12%
of the instances. Evaluations were only used when the evaluators completed all 100 instances for a given ambiguity.



## Citation Information

```
@article{weeber2001developing,
  title    = "Developing a test collection for biomedical word sense
              disambiguation",
  author   = "Weeber, M and Mork, J G and Aronson, A R",
  journal  = "Proc AMIA Symp",
  pages    = "746--750",
  year     =  2001,
  language = "en"
}

```
