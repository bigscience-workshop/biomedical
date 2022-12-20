---
language: en
license: other
multilinguality: monolingual
pretty_name: MedNLI
paperswithcode_id: mednli
---


# Dataset Card for MedNLI

## Dataset Description

- **Homepage:** https://physionet.org/content/mednli/1.0.0/
- **Pubmed:** False
- **Public:** False
- **Tasks:** Textual Entailment


State of the art models using deep neural networks have become very good in learning an accurate
mapping from inputs to outputs. However, they still lack generalization capabilities in conditions
that differ from the ones encountered during training. This is even more challenging in specialized,
and knowledge intensive domains, where training data is limited. To address this gap, we introduce
MedNLI - a dataset annotated by doctors, performing a natural language inference task (NLI),
grounded in the medical history of patients. As the source of premise sentences, we used the
MIMIC-III. More specifically, to minimize the risks to patient privacy, we worked with clinical
notes corresponding to the deceased patients. The clinicians in our team suggested the Past Medical
History to be the most informative section of a clinical note, from which useful inferences can be
drawn about the patient.


## Citation Information

```
@misc{https://doi.org/10.13026/c2rs98,
    title        = {MedNLI — A Natural Language Inference Dataset For The Clinical Domain},
    author       = {Shivade,  Chaitanya},
    year         = 2017,
    publisher    = {physionet.org},
    doi          = {10.13026/C2RS98},
    url          = {https://physionet.org/content/mednli/}
}
```
