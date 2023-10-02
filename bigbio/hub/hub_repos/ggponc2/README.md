---
language:
  - de 
bigbio_language:
  - German
multilinguality: monolingual
pretty_name: GGPONC2
homepage: https://www.leitlinienprogramm-onkologie.de/projekte/ggponc-english/
bigbio_pubmed: false
bigbio_public: flase
bigbio_tasks:
  - NAMED_ENTITY_RECOGNITION
---


# Dataset Card for GGPONC2

## Dataset Description

- **Homepage:** https://www.leitlinienprogramm-onkologie.de/projekte/ggponc-english/
- **Pubmed:** False
- **Public:** False
- **Tasks:** NER 


The GGPONC project aims to provide a freely distributable corpus of German medical text for NLP researchers. 
Clinical guidelines are particularly suitable to create such corpora, as they contain no protected health information 
(PHI), which distinguishes them from other kinds of medical text.

The second version of the corpus (GGPONC 2.0) consists of 30 German oncology guidelines with 1.87 million tokens. 
It has been completely manually annotated on the entity level by 7 medical students using the INCEpTION platform over a 
time frame of 6 months in more than 1200 hours of work. This makes GGPONC 2.0 the largest annotated, freely 
distributable corpus of German medical text at the moment.

Annotated entities are Findings (Diagnosis / Pathology, Other Finding), Substances (Clinical Drug, Nutrients / Body 
Substances, External Substances) and Procedures (Therapeutic, Diagnostic), as well as Specifications for these entities. 
In total, annotators have created more than 200000 entity annotations. In addition, fragment relationships have been 
annotated to explicitly indicate elliptical coordinated noun phrases, a common phenomenon in German text.

## Citation Information

```
@inproceedings{borchert-etal-2022-ggponc,
    title = "{GGPONC} 2.0 - The {G}erman Clinical Guideline Corpus for Oncology: Curation Workflow, Annotation Policy, Baseline {NER} Taggers",
    author = "Borchert, Florian  and
      Lohr, Christina  and
      Modersohn, Luise  and
      Witt, Jonas  and
      Langer, Thomas  and
      Follmann, Markus  and
      Gietzelt, Matthias  and
      Arnrich, Bert  and
      Hahn, Udo  and
      Schapranow, Matthieu-P.",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.389",
    pages = "3650--3660",
}
```
