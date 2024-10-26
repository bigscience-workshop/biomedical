
---
language: 
- en
bigbio_language: 
- English
license: cc-by-4.0
multilinguality: monolingual
bigbio_license_shortname: CC_BY_4p0
pretty_name: ProGene
homepage: https://zenodo.org/record/3698568#.YlVHqdNBxeg
bigbio_pubmed: True
bigbio_public: True
bigbio_tasks: 
- NAMED_ENTITY_RECOGNITION
---


# Dataset Card for ProGene

## Dataset Description

- **Homepage:** https://zenodo.org/record/3698568#.YlVHqdNBxeg
- **Pubmed:** True
- **Public:** True
- **Tasks:** NER


The Protein/Gene corpus was developed at the JULIE Lab Jena under supervision of Prof. Udo Hahn.
The executing scientist was Dr. Joachim Wermter.
The main annotator was Dr. Rico Pusch who is an expert in biology.
The corpus was developed in the context of the StemNet project (http://www.stemnet.de/).



## Citation Information

```
@inproceedings{faessler-etal-2020-progene,
    title = "{P}ro{G}ene - A Large-scale, High-Quality Protein-Gene Annotated Benchmark Corpus",
    author = "Faessler, Erik  and
      Modersohn, Luise  and
      Lohr, Christina  and
      Hahn, Udo",
    booktitle = "Proceedings of the 12th Language Resources and Evaluation Conference",
    month = may,
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2020.lrec-1.564",
    pages = "4585--4596",
    abstract = "Genes and proteins constitute the fundamental entities of molecular genetics. We here introduce ProGene (formerly called FSU-PRGE), a corpus that reflects our efforts to cope with this important class of named entities within the framework of a long-lasting large-scale annotation campaign at the Jena University Language {\&} Information Engineering (JULIE) Lab. We assembled the entire corpus from 11 subcorpora covering various biological domains to achieve an overall subdomain-independent corpus. It consists of 3,308 MEDLINE abstracts with over 36k sentences and more than 960k tokens annotated with nearly 60k named entity mentions. Two annotators strove for carefully assigning entity mentions to classes of genes/proteins as well as families/groups, complexes, variants and enumerations of those where genes and proteins are represented by a single class. The main purpose of the corpus is to provide a large body of consistent and reliable annotations for supervised training and evaluation of machine learning algorithms in this relevant domain. Furthermore, we provide an evaluation of two state-of-the-art baseline systems {---} BioBert and flair {---} on the ProGene corpus. We make the evaluation datasets and the trained models available to encourage comparable evaluations of new methods in the future.",
    language = "English",
    ISBN = "979-10-95546-34-4",
}

```
