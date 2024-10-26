
---
language: 
- en
bigbio_language: 
- English
license: other
multilinguality: monolingual
bigbio_license_shortname: DUA
pretty_name: n2c2 2018 ADE
homepage: https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/
bigbio_pubmed: False
bigbio_public: False
bigbio_tasks: 
- NAMED_ENTITY_RECOGNITION
- RELATION_EXTRACTION
---


# Dataset Card for n2c2 2018 ADE

## Dataset Description

- **Homepage:** https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/
- **Pubmed:** False
- **Public:** False
- **Tasks:** NER,RE


The National NLP Clinical Challenges (n2c2), organized in 2018, continued the
legacy of i2b2 (Informatics for Biology and the Bedside), adding 2 new tracks and 2
new sets of data to the shared tasks organized since 2006. Track 2 of 2018
n2c2 shared tasks focused on the extraction of medications, with their signature
information, and adverse drug events (ADEs) from clinical narratives.
This track built on our previous medication challenge, but added a special focus on ADEs.

ADEs are injuries resulting from a medical intervention related to a drugs and
can include allergic reactions, drug interactions, overdoses, and medication errors.
Collectively, ADEs are estimated to account for 30% of all hospital adverse
events; however, ADEs are preventable. Identifying potential drug interactions,
overdoses, allergies, and errors at the point of care and alerting the caregivers of
potential ADEs can improve health delivery, reduce the risk of ADEs, and improve health
outcomes.

A step in this direction requires processing narratives of clinical records
that often elaborate on the medications given to a patient, as well as the known
allergies, reactions, and adverse events of the patient. Extraction of this information
from narratives complements the structured medication information that can be
obtained from prescriptions, allowing a more thorough assessment of potential ADEs
before they happen.

The 2018 n2c2 shared task Track 2, hereon referred to as the ADE track,
tackled these natural language processing tasks in 3 different steps,
which we refer to as tasks:
1. Concept Extraction: identification of concepts related to medications,
their signature information, and ADEs
2. Relation Classification: linking the previously mentioned concepts to
their medication  by identifying relations on gold standard concepts
3. End-to-End: building end-to-end systems that process raw narrative text
to discover concepts and find relations of those concepts to their medications

Shared tasks provide a venue for head-to-head comparison of systems developed
for the same task and on the same data, allowing researchers to identify the state
of the art in a particular task, learn from it, and build on it.



## Citation Information

```
@article{DBLP:journals/jamia/HenryBFSU20,
  author    = {
                Sam Henry and
                Kevin Buchan and
                Michele Filannino and
                Amber Stubbs and
                Ozlem Uzuner
               },
  title     = {2018 n2c2 shared task on adverse drug events and medication extraction
               in electronic health records},
  journal   = {J. Am. Medical Informatics Assoc.},
  volume    = {27},
  number    = {1},
  pages     = {3--12},
  year      = {2020},
  url       = {https://doi.org/10.1093/jamia/ocz166},
  doi       = {10.1093/jamia/ocz166},
  timestamp = {Sat, 30 May 2020 19:53:56 +0200},
  biburl    = {https://dblp.org/rec/journals/jamia/HenryBFSU20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

```
