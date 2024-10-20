
---
language: 
- en
bigbio_language: 
- English
license: other
multilinguality: monolingual
bigbio_license_shortname: DUA
pretty_name: n2c2 2009 Medications
homepage: https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/
bigbio_pubmed: True
bigbio_public: False
bigbio_tasks: 
- NAMED_ENTITY_RECOGNITION
---


# Dataset Card for n2c2 2009 Medications

## Dataset Description

- **Homepage:** https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/
- **Pubmed:** True
- **Public:** False
- **Tasks:** NER


The Third i2b2 Workshop on Natural Language Processing Challenges for Clinical Records
focused on the identification of medications, their dosages, modes (routes) of administration,
frequencies, durations, and reasons for administration in discharge summaries.
The third i2b2 challenge—that is, the medication challenge—extends information
extraction to relation extraction; it requires extraction of medications and
medication-related information followed by determination of which medication
belongs to which medication-related details.

The medication challenge was designed as an information extraction task.
The goal, for each discharge summary, was to extract the following information
on medications experienced by the patient:
1. Medications (m): including names, brand names, generics, and collective names of prescription substances,
over the counter medications, and other biological substances for which the patient is the experiencer.
2. Dosages (do): indicating the amount of a medication used in each administration.
3. Modes (mo): indicating the route for administering the medication.
4. Frequencies (f): indicating how often each dose of the medication should be taken.
5. Durations (du): indicating how long the medication is to be administered.
6. Reasons (r): stating the medical reason for which the medication is given.
7. Certainty (c): stating whether the event occurs. Certainty can be expressed by uncertainty words,
e.g., “suggested”, or via modals, e.g., “should” indicates suggestion.
8. Event (e): stating on whether the medication is started, stopped, or continued.
9. Temporal (t): stating whether the medication was administered in the past,
is being administered currently, or will be administered in the future, to the extent
that this information is expressed in the tense of the verbs and auxiliary verbs used to express events.
10. List/narrative (ln): indicating whether the medication information appears in a
list structure or in narrative running text in the discharge summary.

The medication challenge asked that systems extract the text corresponding to each of the fields
for each of the mentions of the medications that were experienced by the patients.

The values for the set of fields related to a medication mention, if presented within a
two-line window of the mention, were linked in order to create what we defined as an ‘entry’.
If the value of a field for a mention were not specified within a two-line window,
then the value ‘nm’ for ‘not mentioned’ was entered and the offsets were left unspecified.

Since the dataset annotations were crowd-sourced, it contains various violations that are handled
throughout the data loader via means of exception catching or conditional statements. e.g.
annotation: anticoagulation, while in text all words are to be separated by space which
means words at end of sentence will always contain `.` and hence won't be an exact match
i.e. `anticoagulation` != `anticoagulation.` from doc_id: 818404



## Citation Information

```
@article{DBLP:journals/jamia/UzunerSC10,
  author    = {
                Ozlem Uzuner and
                Imre Solti and
                Eithon Cadag
               },
  title     = {Extracting medication information from clinical text},
  journal   = {J. Am. Medical Informatics Assoc.},
  volume    = {17},
  number    = {5},
  pages     = {514--518},
  year      = {2010},
  url       = {https://doi.org/10.1136/jamia.2010.003947},
  doi       = {10.1136/jamia.2010.003947},
  timestamp = {Mon, 11 May 2020 22:59:55 +0200},
  biburl    = {https://dblp.org/rec/journals/jamia/UzunerSC10.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

```
