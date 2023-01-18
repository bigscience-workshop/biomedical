
---
language: 
- es
bigbio_language: 
- Spanish
license: cc-by-4.0
multilinguality: monolingual
bigbio_license_shortname: CC_BY_4p0
pretty_name: MESINESP 2021
homepage: https://zenodo.org/record/5602914#.YhSXJ5PMKWt
bigbio_pubmed: False
bigbio_public: True
bigbio_tasks: 
- TEXT_CLASSIFICATION
---


# Dataset Card for MESINESP 2021

## Dataset Description

- **Homepage:** https://zenodo.org/record/5602914#.YhSXJ5PMKWt
- **Pubmed:** False
- **Public:** True
- **Tasks:** TXTCLASS


The main aim of MESINESP2 is to promote the development of practically relevant semantic indexing tools for biomedical content in non-English language. We have generated a manually annotated corpus, where domain experts have labeled a set of scientific literature, clinical trials, and patent abstracts. All the documents were labeled with DeCS descriptors, which is a structured controlled vocabulary created by BIREME to index scientific publications on BvSalud, the largest database of scientific documents in Spanish, which hosts records from the databases LILACS, MEDLINE, IBECS, among others.

MESINESP track at BioASQ9 explores the efficiency of systems for assigning DeCS to different types of biomedical documents. To that purpose, we have divided the task into three subtracks depending on the document type. Then, for each one we generated an annotated corpus which was provided to participating teams:

- [Subtrack 1 corpus] MESINESP-L – Scientific Literature: It contains all   Spanish records from LILACS and IBECS databases at the Virtual Health Library   (VHL) with non-empty abstract written in Spanish.
- [Subtrack 2 corpus] MESINESP-T- Clinical Trials contains records from Registro   Español de Estudios Clínicos (REEC). REEC doesn't provide documents with the   structure title/abstract needed in BioASQ, for that reason we have built   artificial abstracts based on the content available in the data crawled using   the REEC API.
- [Subtrack 3 corpus] MESINESP-P – Patents: This corpus includes patents in   Spanish extracted from Google Patents which have the IPC code “A61P” and   “A61K31”. In addition, we also provide a set of complementary data such as:   the DeCS terminology file, a silver standard with the participants' predictions   to the task background set and the entities of medications, diseases, symptoms   and medical procedures extracted from the BSC NERs documents.



## Citation Information

```
@conference {396,
    title = {Overview of BioASQ 2021-MESINESP track. Evaluation of
    advance hierarchical classification techniques for scientific
    literature, patents and clinical trials.},
    booktitle = {Proceedings of the 9th BioASQ Workshop
    A challenge on large-scale biomedical semantic indexing
    and question answering},
    year = {2021},
    url = {http://ceur-ws.org/Vol-2936/paper-11.pdf},
    author = {Gasco, Luis and Nentidis, Anastasios and Krithara, Anastasia
     and Estrada-Zavala, Darryl and Toshiyuki Murasaki, Renato and Primo-Pe{\~n}a,
     Elena and Bojo-Canales, Cristina and Paliouras, Georgios and Krallinger, Martin}
}


```
