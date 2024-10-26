
---
language: 
- en
bigbio_language: 
- English
license: other
multilinguality: monolingual
bigbio_license_shortname: DUA
pretty_name: n2c2 2006 De-identification
homepage: https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/
bigbio_pubmed: False
bigbio_public: False
bigbio_tasks: 
- NAMED_ENTITY_RECOGNITION
---


# Dataset Card for n2c2 2006 De-identification

## Dataset Description

- **Homepage:** https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/
- **Pubmed:** False
- **Public:** False
- **Tasks:** NER


The data for the de-identification challenge came from Partners Healthcare and
included solely medical discharge summaries. We prepared the data for the
challengeby annotating and by replacing all authentic PHI with realistic
surrogates.

Given the above definitions, we marked the authentic PHI in the records in two stages.
In the first stage, we used an automatic system.31 In the second stage, we validated
the output of the automatic system manually. Three annotators, including undergraduate
and graduate students and a professor, serially made three manual passes over each record.
They marked and discussed the PHI tags they disagreed on and finalized these tags
after discussion.

The original dataset does not have spans for each entity. The spans are
computed in this loader and the final text that correspond with the
tags is preserved  in the source format



## Citation Information

```
@article{uzuner2007evaluating,
    author = {
        Uzuner, Özlem and
        Luo, Yuan and
        Szolovits, Peter
    },
    title     = {Evaluating the State-of-the-Art in Automatic De-identification},
    journal   = {Journal of the American Medical Informatics Association},
    volume    = {14},
    number    = {5},
    pages     = {550-563},
    year      = {2007},
    month     = {09},
    url       = {https://doi.org/10.1197/jamia.M2444},
    doi       = {10.1197/jamia.M2444},
    eprint    = {https://academic.oup.com/jamia/article-pdf/14/5/550/2136261/14-5-550.pdf}
}

```
