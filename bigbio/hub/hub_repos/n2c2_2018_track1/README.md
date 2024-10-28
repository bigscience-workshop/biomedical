
---
language: 
- en
bigbio_language: 
- English
license: other
multilinguality: monolingual
bigbio_license_shortname: DUA
pretty_name: n2c2 2018 Selection Criteria
homepage: https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/
bigbio_pubmed: False
bigbio_public: False
bigbio_tasks: 
- TEXT_CLASSIFICATION
---


# Dataset Card for n2c2 2018 Selection Criteria

## Dataset Description

- **Homepage:** https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/
- **Pubmed:** False
- **Public:** False
- **Tasks:** TXTCLASS


Track 1 of the 2018 National NLP Clinical Challenges shared tasks focused
on identifying which patients in a corpus of longitudinal medical records
meet and do not meet identified selection criteria.

This shared task aimed to determine whether NLP systems could be trained to identify if patients met or did not meet
a set of selection criteria taken from real clinical trials. The selected criteria required measurement detection (
“Any HbA1c value between 6.5 and 9.5%”), inference (“Use of aspirin to prevent myocardial infarction”),
temporal reasoning (“Diagnosis of ketoacidosis in the past year”), and expert judgment to assess (“Major
diabetes-related complication”). For the corpus, we used the dataset of American English, longitudinal clinical
narratives from the 2014 i2b2/UTHealth shared task 4.

The final selected 13 selection criteria are as follows:
1. DRUG-ABUSE: Drug abuse, current or past
2. ALCOHOL-ABUSE: Current alcohol use over weekly recommended limits
3. ENGLISH: Patient must speak English
4. MAKES-DECISIONS: Patient must make their own medical decisions
5. ABDOMINAL: History of intra-abdominal surgery, small or large intestine
resection, or small bowel obstruction.
6. MAJOR-DIABETES: Major diabetes-related complication. For the purposes of
this annotation, we define “major complication” (as opposed to “minor complication”)
as any of the following that are a result of (or strongly correlated with) uncontrolled diabetes:
    a. Amputation
    b. Kidney damage
    c. Skin conditions
    d. Retinopathy
    e. nephropathy
    f. neuropathy
7. ADVANCED-CAD: Advanced cardiovascular disease (CAD).
For the purposes of this annotation, we define “advanced” as having 2 or more of the following:
    a. Taking 2 or more medications to treat CAD
    b. History of myocardial infarction (MI)
    c. Currently experiencing angina
    d. Ischemia, past or present
8. MI-6MOS: MI in the past 6 months
9. KETO-1YR: Diagnosis of ketoacidosis in the past year
10. DIETSUPP-2MOS: Taken a dietary supplement (excluding vitamin D) in the past 2 months
11. ASP-FOR-MI: Use of aspirin to prevent MI
12. HBA1C: Any hemoglobin A1c (HbA1c) value between 6.5% and 9.5%
13. CREATININE: Serum creatinine > upper limit of normal

The training consists of 202 patient records with document-level annotations, 10 records
with textual spans indicating annotator’s evidence for their annotations while test set contains 86.

Note:
* The inter-annotator average agreement is 84.9%
* Whereabouts of 10 records with textual spans indicating annotator’s evidence are unknown.
However, author did a simple script based validation to check if any of the tags contained any text
in any of the training set and they do not, which confirms that atleast train and test do not
 have any evidence tagged alongside corresponding tags.



## Citation Information

```
@article{DBLP:journals/jamia/StubbsFSHU19,
  author    = {
                Amber Stubbs and
                Michele Filannino and
                Ergin Soysal and
                Samuel Henry and
                Ozlem Uzuner
               },
  title     = {Cohort selection for clinical trials: n2c2 2018 shared task track 1},
  journal   = {J. Am. Medical Informatics Assoc.},
  volume    = {26},
  number    = {11},
  pages     = {1163--1171},
  year      = {2019},
  url       = {https://doi.org/10.1093/jamia/ocz163},
  doi       = {10.1093/jamia/ocz163},
  timestamp = {Mon, 15 Jun 2020 16:56:11 +0200},
  biburl    = {https://dblp.org/rec/journals/jamia/StubbsFSHU19.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

```
