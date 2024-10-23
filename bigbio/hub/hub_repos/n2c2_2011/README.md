
---
language: 
- en
bigbio_language: 
- English
license: other
multilinguality: monolingual
bigbio_license_shortname: DUA
pretty_name: n2c2 2011 Coreference
homepage: https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/
bigbio_pubmed: False
bigbio_public: False
bigbio_tasks: 
- COREFERENCE_RESOLUTION
---


# Dataset Card for n2c2 2011 Coreference

## Dataset Description

- **Homepage:** https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/
- **Pubmed:** False
- **Public:** False
- **Tasks:** COREF


The i2b2/VA corpus contained de-identified discharge summaries from Beth Israel
Deaconess Medical Center, Partners Healthcare, and University of Pittsburgh Medical
Center (UPMC). In addition, UPMC contributed de-identified progress notes to the
i2b2/VA corpus. This dataset contains the records from Beth Israel and Partners.

The i2b2/VA corpus contained five concept categories: problem, person, pronoun,
test, and treatment. Each record in the i2b2/VA corpus was annotated by two
independent annotators for coreference pairs. Then the pairs were post-processed
in order to create coreference chains. These chains were presented to an adjudicator,
who resolved the disagreements between the original annotations, and added or deleted
annotations as necessary. The outputs of the adjudicators were then re-adjudicated, with
particular attention being paid to duplicates and enforcing consistency in the annotations.




## Citation Information

```
@article{uzuner2012evaluating,
    author = {
        Uzuner, Ozlem and
        Bodnari, Andreea and
        Shen, Shuying and
        Forbush, Tyler and
        Pestian, John and
        South, Brett R
    },
    title = "{Evaluating the state of the art in coreference resolution for electronic medical records}",
    journal = {Journal of the American Medical Informatics Association},
    volume = {19},
    number = {5},
    pages = {786-791},
    year = {2012},
    month = {02},
    issn = {1067-5027},
    doi = {10.1136/amiajnl-2011-000784},
    url = {https://doi.org/10.1136/amiajnl-2011-000784},
    eprint = {https://academic.oup.com/jamia/article-pdf/19/5/786/17374287/19-5-786.pdf},
}

```
