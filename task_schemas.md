# BigBio Schema Documentation
We have defined a set of lightwieght, task-specific schema to help simplify programmatic access to common biomedical datasets. This schema should be implemented for each dataset in addition to a schema that preserves the original dataset format.

### Example Schemas by Task
- Information Extraction
 - Named entity recognition (NER)
 - Named entity disambiguation/normalization/linking (NED)
 - Event extraction
 - Relation extraction (RE)
 - Coreference resolution
- Textual Entailment (TE)
- Question Answering (QA)
- Paraphasing, Translation
- Text Classification 

## Information Extraction

[Schema Template](foo)

This is a simple container format with minimal nesting that supports a range of common information extraction/knowledge base construction tasks.

- Named entity recognition (NER)
- Named entity disambiguation/normalization/linking (NED)
- Event extraction
- Relation extraction (RE)
- Coreference resolution

```
{
    "document_id": "XXXXXX",
    "passages": [...],
    "entities": [...],
    "events": [...],
    "coreferences": [...],
    "relations": [...]
}
```



**Schema Notes**

- `id` must be unique integer within dataset and across parent category (e.g., entities, passages)
- `offsets` contains absolute character offsets per span
- `offsets` and `text` are lists to support discontinous spans. 
- `normalized` may contain 1 or more normalized links to database entity identifiers.
- `passages` captures document structure such as named sections.
- `entities`,`events`,`coreferences`,`relations` may be empty fields based on dataset and specific task.



### Passages

Passages capture document structure, such as the title and abstact sections of a PubMed abstract. 

```
{
    "document_id": "227508",
    "passages": [
        {
            "id": 0,
            "type": "title",
            "text": "Naloxone reverses the antihypertensive effect of clonidine.",
            "offsets": [[0, 59]],
        },
        {
            "id": 1,
            "type": "abstract",
            "text": "In unanesthetized, spontaneously hypertensive rats the decrease in blood pressure and heart rate produced by intravenous clonidine, 5 to 20 micrograms/kg, was inhibited or reversed by nalozone, 0.2 to 2 mg/kg. The hypotensive effect of 100 mg/kg alpha-methyldopa was also partially reversed by naloxone. Naloxone alone did not affect either blood pressure or heart rate. In brain membranes from spontaneously hypertensive rats clonidine, 10(-8) to 10(-5) M, did not influence stereoselective binding of [3H]-naloxone (8 nM), and naloxone, 10(-8) to 10(-4) M, did not influence clonidine-suppressible binding of [3H]-dihydroergocryptine (1 nM). These findings indicate that in spontaneously hypertensive rats the effects of central alpha-adrenoceptor stimulation involve activation of opiate receptors. As naloxone and clonidine do not appear to interact with the same receptor site, the observed functional antagonism suggests the release of an endogenous opiate by clonidine or alpha-methyldopa and the possible role of the opiate in the central control of sympathetic tone.",
            "offsets": [[60, 1075]],
        },
    ],
}
```

### Entities

- Examples: [BC5CDR](), [NCBI Disease]()

```
"entities": [
    {
        "id": 2,
        "offsets": [[0, 8]],
        "text": ["Naloxone"],
        "type": "Chemical",
        "normalized": [{"db_name": "MESH", "db_id": "D009270"}]
    },
    ...
 ],
```

### Events

### Coreferences

- Examples: [n2c2 2011: Coreference Challenge](https://huggingface.co/datasets/scitail)

```
"coreferences": [
	{
   		"id": 32,
   		"entity_ids": [1, 10, 23],
	},
	...
]
```

### Relations
- Examples: [BC5CDR]()

```
"relations": [
    {
        "id": 100,
        "type": "chemical-induced disease",
        "arg1_id": 10,
        "arg2_id": 32,
        "normalized": []
    }
]
```

## Textual Entailment (TE)

- [Schema Template](foo)
- Examples: [SciTail](https://huggingface.co/datasets/scitail)


```
{
	"id": 0,
    "premise": "Pluto rotates once on its axis every 6.39 Earth days;",
    "hypothesis": "Earth rotates on its axis once times in one day.",
    "label": "neutral",
}
```

## Question Answering (QA)
- [Schema Template](foo)
- Examples: [BioASQ9 Task B](https://huggingface.co/datasets/scitail)

```
{
    "id": record["id"],
    "document_id": "",
    "type": record["body"],
    "question": "Is Hirschsprung disease a mendelian or a multifactorial disorder?", 
    "context": "Hirschsprung disease (HSCR) is a multifactorial, non-mendelian disorder in which rare high-penetrance coding sequence mutations in the receptor tyrosine kinase RET contribute to risk in combination with mutations at other genes", 
    "answer": record["body"],
    "expanded_answer": record["body"],
}
```