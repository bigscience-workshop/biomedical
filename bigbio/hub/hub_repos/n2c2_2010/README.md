
---
language: 
- en
bigbio_language: 
- English
license: other
multilinguality: monolingual
bigbio_license_shortname: DUA
pretty_name: n2c2 2010 Concepts, Assertions, and Relations
homepage: https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/
bigbio_pubmed: False
bigbio_public: False
bigbio_tasks: 
- NAMED_ENTITY_RECOGNITION
- RELATION_EXTRACTION
---


# Dataset Card for n2c2 2010 Concepts, Assertions, and Relations

## Dataset Description

- **Homepage:** https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/
- **Pubmed:** False
- **Public:** False
- **Tasks:** NER,RE


The i2b2/VA corpus contained de-identified discharge summaries from Beth Israel
Deaconess Medical Center, Partners Healthcare, and University of Pittsburgh Medical
Center (UPMC). In addition, UPMC contributed de-identified progress notes to the
i2b2/VA corpus. This dataset contains the records from Beth Israel and Partners.

The 2010 i2b2/VA Workshop on Natural Language Processing Challenges for Clinical Records comprises three tasks:
1) a concept extraction task focused on the extraction of medical concepts from patient reports;
2) an assertion classification task focused on assigning assertion types for medical problem concepts;
3) a relation classification task focused on assigning relation types that hold between medical problems,
tests, and treatments.

i2b2 and the VA provided an annotated reference standard corpus for the three tasks.
Using this reference standard, 22 systems were developed for concept extraction,
21 for assertion classification, and 16 for relation classification.



## Citation Information

```
@article{DBLP:journals/jamia/UzunerSSD11,
  author    = {
                Ozlem Uzuner and
                Brett R. South and
                Shuying Shen and
                Scott L. DuVall
               },
  title     = {2010 i2b2/VA challenge on concepts, assertions, and relations in clinical
               text},
  journal   = {J. Am. Medical Informatics Assoc.},
  volume    = {18},
  number    = {5},
  pages     = {552--556},
  year      = {2011},
  url       = {https://doi.org/10.1136/amiajnl-2011-000203},
  doi       = {10.1136/amiajnl-2011-000203},
  timestamp = {Mon, 11 May 2020 23:00:20 +0200},
  biburl    = {https://dblp.org/rec/journals/jamia/UzunerSSD11.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

```
