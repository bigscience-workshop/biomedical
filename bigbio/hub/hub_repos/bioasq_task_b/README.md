---
language: en
license: other
multilinguality: monolingual
pretty_name: BioASQ Task B
---


# Dataset Card for BioASQ Task B

## Dataset Description

- **Homepage:** http://participants-area.bioasq.org/datasets/
- **Pubmed:** True
- **Public:** False
- **Tasks:** Question Answering


The BioASQ corpus contains multiple question
answering tasks annotated by biomedical experts, including yes/no, factoid, list,
and summary questions. Pertaining to our objective of comparing neural language
models, we focus on the the yes/no questions (Task 7b), and leave the inclusion
of other tasks to future work. Each question is paired with a reference text
containing multiple sentences from a PubMed abstract and a yes/no answer. We use
the official train/dev/test split of 670/75/140 questions.

See 'Domain-Specific Language Model Pretraining for Biomedical
Natural Language Processing'


## Citation Information

```
@article{tsatsaronis2015overview,
	title        = {
		An overview of the BIOASQ large-scale biomedical semantic indexing and
		question answering competition
	},
	author       = {
		Tsatsaronis, George and Balikas, Georgios and Malakasiotis, Prodromos
        and Partalas, Ioannis and Zschunke, Matthias and Alvers, Michael R and
		Weissenborn, Dirk and Krithara, Anastasia and Petridis, Sergios and
		Polychronopoulos, Dimitris and others
	},
	year         = 2015,
	journal      = {BMC bioinformatics},
	publisher    = {BioMed Central Ltd},
	volume       = 16,
	number       = 1,
	pages        = 138
}
```
