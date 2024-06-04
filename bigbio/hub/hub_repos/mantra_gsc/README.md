---
language:
  - en, fr, de, nl, es
bigbio_language:
  - English, French, German, Dutch, Spanish
license: gpl-3.0
bigbio_license_shortname: GPL_3p0_ONLY
multilinguality: multilingual
pretty_name: MantraGSC
homepage: https://github.com/mi-erasmusmc/Mantra-Gold-Standard-Corpus
bigbio_pubmed: true
bigbio_public: true
bigbio_tasks:
  - NAMED_ENTITY_RECOGNITION
  - NAMED_ENTITY_DISAMBIGUATION
---


# Dataset Card for Mantra GSC

## Dataset Description

- **Homepage:** https://github.com/mi-erasmusmc/Mantra-Gold-Standard-Corpus
- **Pubmed:** True
- **Public:** True
- **Tasks:** NER, NED

We selected text units from different parallel corpora (Medline abstract titles, drug labels, biomedical patent claims) in English, French, German, Spanish, and Dutch. Three annotators per language independently annotated the biomedical concepts, based on a subset of the Unified Medical Language System and covering a wide range of semantic groups.

## Citation Information

```
@article{10.1093/jamia/ocv037,
    author = {Kors, Jan A and Clematide, Simon and Akhondi,
    Saber A and van Mulligen, Erik M and Rebholz-Schuhmann, Dietrich},
    title = "{A multilingual gold-standard corpus for biomedical concept recognition: the Mantra GSC}",
    journal = {Journal of the American Medical Informatics Association},
    volume = {22},
    number = {5},
    pages = {948-956},
    year = {2015},
    month = {05},
    abstract = "{Objective To create a multilingual gold-standard corpus for biomedical concept recognition.Materials
    and methods We selected text units from different parallel corpora (Medline abstract titles, drug labels,
    biomedical patent claims) in English, French, German, Spanish, and Dutch. Three annotators per language
    independently annotated the biomedical concepts, based on a subset of the Unified Medical Language System and
    covering a wide range of semantic groups. To reduce the annotation workload, automatically generated
    preannotations were provided. Individual annotations were automatically harmonized and then adjudicated, and
    cross-language consistency checks were carried out to arrive at the final annotations.Results The number of final
    annotations was 5530. Inter-annotator agreement scores indicate good agreement (median F-score 0.79), and are
    similar to those between individual annotators and the gold standard. The automatically generated harmonized
    annotation set for each language performed equally well as the best annotator for that language.Discussion The use
    of automatic preannotations, harmonized annotations, and parallel corpora helped to keep the manual annotation
    efforts manageable. The inter-annotator agreement scores provide a reference standard for gauging the performance
    of automatic annotation techniques.Conclusion To our knowledge, this is the first gold-standard corpus for
    biomedical concept recognition in languages other than English. Other distinguishing features are the wide variety
    of semantic groups that are being covered, and the diversity of text genres that were annotated.}",
    issn = {1067-5027},
    doi = {10.1093/jamia/ocv037},
    url = {https://doi.org/10.1093/jamia/ocv037},
    eprint = {https://academic.oup.com/jamia/article-pdf/22/5/948/34146393/ocv037.pdf},
}
```
