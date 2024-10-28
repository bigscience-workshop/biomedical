
---
language: 
- en
bigbio_language: 
- English
license: cc-by-nc-3.0
multilinguality: monolingual
bigbio_license_shortname: CC_BY_NC_3p0
pretty_name: miRNA
homepage: https://www.scai.fraunhofer.de/en/business-research-areas/bioinformatics/downloads/download-mirna-test-corpus.html
bigbio_pubmed: True
bigbio_public: True
bigbio_tasks: 
- NAMED_ENTITY_RECOGNITION
- NAMED_ENTITY_DISAMBIGUATION
---


# Dataset Card for miRNA

## Dataset Description

- **Homepage:** https://www.scai.fraunhofer.de/en/business-research-areas/bioinformatics/downloads/download-mirna-test-corpus.html
- **Pubmed:** True
- **Public:** True
- **Tasks:** NER,NED


The corpus consists of 301 Medline citations. The documents were screened for
mentions of miRNA in the abstract text. Gene, disease and miRNA entities were manually
annotated. The corpus comprises of two separate files, a train and a test set, coming
from 201 and 100 documents respectively. 



## Citation Information

```
@Article{Bagewadi2014,
author={Bagewadi, Shweta
and Bobi{'{c}}, Tamara
and Hofmann-Apitius, Martin
and Fluck, Juliane
and Klinger, Roman},
title={Detecting miRNA Mentions and Relations in Biomedical Literature},
journal={F1000Research},
year={2014},
month={Aug},
day={28},
publisher={F1000Research},
volume={3},
pages={205-205},
keywords={MicroRNAs; corpus; prediction algorithms},
abstract={
    INTRODUCTION: MicroRNAs (miRNAs) have demonstrated their potential as post-transcriptional
    gene expression regulators, participating in a wide spectrum of regulatory events such as
    apoptosis, differentiation, and stress response. Apart from the role of miRNAs in normal
    physiology, their dysregulation is implicated in a vast array of diseases. Dissection of
    miRNA-related associations are valuable for contemplating their mechanism in diseases,
    leading to the discovery of novel miRNAs for disease prognosis, diagnosis, and therapy.
    MOTIVATION: Apart from databases and prediction tools, miRNA-related information is largely
    available as unstructured text. Manual retrieval of these associations can be labor-intensive
    due to steadily growing number of publications. Additionally, most of the published miRNA
    entity recognition methods are keyword based, further subjected to manual inspection for
    retrieval of relations. Despite the fact that several databases host miRNA-associations
    derived from text, lower sensitivity and lack of published details for miRNA entity
    recognition and associated relations identification has motivated the need for developing
    comprehensive methods that are freely available for the scientific community. Additionally,
    the lack of a standard corpus for miRNA-relations has caused difficulty in evaluating the
    available systems. We propose methods to automatically extract mentions of miRNAs, species,
    genes/proteins, disease, and relations from scientific literature. Our generated corpora,
    along with dictionaries, and miRNA regular expression are freely available for academic
    purposes. To our knowledge, these resources are the most comprehensive developed so far.
    RESULTS: The identification of specific miRNA mentions reaches a recall of 0.94 and
    precision of 0.93. Extraction of miRNA-disease and miRNA-gene relations lead to an
    F1 score of up to 0.76. A comparison of the information extracted by our approach to
    the databases miR2Disease and miRSel for the extraction of Alzheimer's disease
    related relations shows the capability of our proposed methods in identifying correct
    relations with improved sensitivity. The published resources and described methods can
    help the researchers for maximal retrieval of miRNA-relations and generation of
    miRNA-regulatory networks. AVAILABILITY: The training and test corpora, annotation
    guidelines, developed dictionaries, and supplementary files are available at
    http://www.scai.fraunhofer.de/mirna-corpora.html.
},
note={26535109[pmid]},
note={PMC4602280[pmcid]},
issn={2046-1402},
url={https://pubmed.ncbi.nlm.nih.gov/26535109},
language={eng}
}

```
