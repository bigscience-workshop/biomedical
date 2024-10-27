
---
language: 
- en
bigbio_language: 
- English
license: mit
multilinguality: monolingual
bigbio_license_shortname: MIT
pretty_name: Evidence Inference 2.0
homepage: https://github.com/jayded/evidence-inference
bigbio_pubmed: True
bigbio_public: True
bigbio_tasks: 
- QUESTION_ANSWERING
---


# Dataset Card for Evidence Inference 2.0

## Dataset Description

- **Homepage:** https://github.com/jayded/evidence-inference
- **Pubmed:** True
- **Public:** True
- **Tasks:** QA


The dataset consists of biomedical articles describing randomized control trials (RCTs) that compare multiple
treatments. Each of these articles will have multiple questions, or 'prompts' associated with them.
These prompts will ask about the relationship between an intervention and comparator with respect to an outcome,
as reported in the trial. For example, a prompt may ask about the reported effects of aspirin as compared
to placebo on the duration of headaches. For the sake of this task, we assume that a particular article
will report that the intervention of interest either significantly increased, significantly decreased
or had significant effect on the outcome, relative to the comparator.



## Citation Information

```
@inproceedings{deyoung-etal-2020-evidence,
    title = "Evidence Inference 2.0: More Data, Better Models",
    author = "DeYoung, Jay  and
      Lehman, Eric  and
      Nye, Benjamin  and
      Marshall, Iain  and
      Wallace, Byron C.",
    booktitle = "Proceedings of the 19th SIGBioMed Workshop on Biomedical Language Processing",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.bionlp-1.13",
    pages = "123--132",
}

```
