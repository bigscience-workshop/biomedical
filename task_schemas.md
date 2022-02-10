# BigBio Schema Documentation
We have defined a set of lightwieght, task-specific schema to help simplify programmatic access to common biomedical datasets. This schema should be implemented for each dataset in addition to a schema that preserves the original dataset format.

### Example Schemas by Task
- [Information Extraction](#information-extraction)
  - Named entity recognition (NER)
  - Named entity disambiguation/normalization/linking (NED)
  - Event extraction
  - Relation extraction (RE)
  - Coreference resolution
- [Textual Entailment (TE)](#textual-entailment)
- [Question Answering (QA)](#question-answering)
- [Paraphasing, Translation, Summarization](#translation,-paraphasing,-summarization)
- [Semantic Similarity](#semantic-similarity)
- [Text Classification](#text-classification)

## Information Extraction

[Schema Template](schemas/kb.py)

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
    "id": 0,
    "document_id": "227508",
    "passages": [
        {
            "id": 1,
            "type": "title",
            "text": "Naloxone reverses the antihypertensive effect of clonidine.",
            "offsets": [[0, 59]],
        },
        {
            "id": 2,
            "type": "abstract",
            "text": "In unanesthetized, spontaneously hypertensive rats the decrease in blood pressure and heart rate produced by intravenous clonidine, 5 to 20 micrograms/kg, was inhibited or reversed by nalozone, 0.2 to 2 mg/kg. The hypotensive effect of 100 mg/kg alpha-methyldopa was also partially reversed by naloxone. Naloxone alone did not affect either blood pressure or heart rate. In brain membranes from spontaneously hypertensive rats clonidine, 10(-8) to 10(-5) M, did not influence stereoselective binding of [3H]-naloxone (8 nM), and naloxone, 10(-8) to 10(-4) M, did not influence clonidine-suppressible binding of [3H]-dihydroergocryptine (1 nM). These findings indicate that in spontaneously hypertensive rats the effects of central alpha-adrenoceptor stimulation involve activation of opiate receptors. As naloxone and clonidine do not appear to interact with the same receptor site, the observed functional antagonism suggests the release of an endogenous opiate by clonidine or alpha-methyldopa and the possible role of the opiate in the central control of sympathetic tone.",
            "offsets": [[60, 1075]],
        },
    ],
}
```

### Entities

- Examples: [BC5CDR](examples/bc5cdr.py)

```
"entities": [
    {
        "id": 3,
        "offsets": [[0, 8]],
        "text": ["Naloxone"],
        "type": "Chemical",
        "normalized": [{"db_name": "MESH", "db_id": "D009270"}]
    },
    ...
 ],
```

### Events
- Examples: [MLEE]()

```
TBD
```

### Coreferences

- Examples: [n2c2 2011: Coreference Challenge]()

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
- Examples: [BC5CDR](examples/bc5cdr.py)

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

## Question Answering (QA)
- [Schema Template](schemas/qa.py)
- Examples: [BioASQ9 Task B](examples/bioasq9b.py)

```
{
	"id": 0,
	"document_id": "24267510",
	"question_id": "55031181e9bde69634000014",
	"question": "Is RANKL secreted from the cells?", 
	"type": "yesno",
	"context": Osteoprotegerin (OPG) is a soluble secreted factor that acts as a decoy receptor for receptor activator of NF-\u03baB ligand (RANKL)", 
	"answer": "yes",
}
```

## Textual Entailment (TE)

- [Schema Template](schemas/entailment.py)
- Examples: [SciTail](examples/scitail.py)

```
{
	"id": 0,
	"document_id": "NULL",
	"premise": "Pluto rotates once on its axis every 6.39 Earth days;",
	"hypothesis": "Earth rotates on its axis once times in one day.",
	"label": "neutral",
}
```

## Translation, Paraphasing, Summarization

- [Schema Template](schema/text_to_text.py)
- Examples: [ParaMed](examples/paramed.py)

```
{
	"id": 0,
	"document_id": "NULL",
	"text_1": "也许 不能 : 分析 结果 提示 激素 疗法 在 维持 去 脂 体重 方面 作用 很小 .",
	"text_2": "probably not : analysis suggests minimal effect of HT in maintaining lean body mass .",
	"text_1_name": "zh",
	"text_2_name": "en",
}
```


## Semantic Similarity
- [Schema Template](schema/pairs.py)
- Examples: [MQP](examples/mqp.py)

```
{
	"id": 0,
	"document_id": "NULL",
	"text_1": "Am I over weight (192.9) for my age (39)?",
	"text_2": "I am a 39 y/o male currently weighing about 193 lbs. Do you think I am overweight?",
	"label": 1,
}
```

## Text Classification
- [Schema Template](schema/text.py)

```
{
	"id": 0,
	"document_id": "NULL",
	"text": "Am I over weight (192.9) for my age (39)?",
	"label": 1,
}
```