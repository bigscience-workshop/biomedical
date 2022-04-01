# BigBio Schema Documentation
We have defined a set of lightwieght, task-specific schema to help simplify programmatic access to common biomedical datasets. This schema should be implemented for each dataset in addition to a schema that preserves the original dataset format.

### Example Schema and Associated Tasks

- [Knowledge Base (KB)](#knowledge-base)
  - Named entity recognition (NER)
  - Named entity disambiguation/normalization/linking (NED)
  - Event extraction (EE)
  - Relation extraction (RE)
  - Coreference resolution (COREF)
- [Question Answering (QA)](#question-answering)
  - Question answering (QA)
- [Textual Entailment (TE)](#textual-entailment)
  - Textual entailment (TE)
- [Text Pairs (PAIRS)](#text-pairs)
  - Semantic Similarity (STS)
- [Text to Text (T2T)](#text-to-text)
  - Paraphasing (PARA)
  - Translation (TRANSL)
  - Summarization (SUM)
- [Text (TEXT)](#text)
  - Text classification (TXTCLASS)


## Knowledge Base

[Schema Template](utils/schemas/kb.py)

This is a simple container format with minimal nesting that supports a range of common knowledge base construction / information extraction tasks.

- Named entity recognition (NER)
- Named entity disambiguation/normalization/linking (NED)
- Event extraction (EE)
- Relation extraction (RE)
- Coreference resolution (COREF)

```
{
    "id": "ABCDEFG",
    "document_id": "XXXXXX",
    "passages": [...],
    "entities": [...],
    "events": [...],
    "coreferences": [...],
    "relations": [...]
}
```



**Schema Notes**

- `id` fields appear at the top (i.e. document) level and in every sub-component (`passages`, `entities`, `events`, `coreferences`, `relations`). They can be set in any fashion that makes every `id` field in a dataset unique (including `id` fields in different splits like train/validation/test).
- `document_id` should be a dataset provided document id. If not provided in the dataset, it can be set equal to the top level `id`.
- `offsets` contain character offsets into the string that would be created from `" ".join([passage["text"] for passage in passages])`
- `offsets` and `text` are always lists to support discontinous spans. For continuous spans, they will have the form `offsets=[(lo,hi)], text=["text span"]`. For discontinuous spans, they will have the form `offsets=[(lo1,hi1), (lo2,hi2), ...], text=["text span 1", "text span 2", ...]`
- `normalized` sub-component may contain 1 or more normalized links to database entity identifiers.
- `passages` captures document structure such as named sections.
- `entities`,`events`,`coreferences`,`relations` may be empty fields depending on the dataset and specific task.



### Passages

Passages capture document structure, such as the title and abstact sections of a PubMed abstract.

```
{
    "id": "0",
    "document_id": "227508",
    "passages": [
        {
            "id": "1",
            "type": "title",
            "text": ["Naloxone reverses the antihypertensive effect of clonidine."],
            "offsets": [[0, 59]],
        },
        {
            "id": "2",
            "type": "abstract",
            "text": ["In unanesthetized, spontaneously hypertensive rats the decrease in blood pressure and heart rate produced by intravenous clonidine, 5 to 20 micrograms/kg, was inhibited or reversed by nalozone, 0.2 to 2 mg/kg. The hypotensive effect of 100 mg/kg alpha-methyldopa was also partially reversed by naloxone. Naloxone alone did not affect either blood pressure or heart rate. In brain membranes from spontaneously hypertensive rats clonidine, 10(-8) to 10(-5) M, did not influence stereoselective binding of [3H]-naloxone (8 nM), and naloxone, 10(-8) to 10(-4) M, did not influence clonidine-suppressible binding of [3H]-dihydroergocryptine (1 nM). These findings indicate that in spontaneously hypertensive rats the effects of central alpha-adrenoceptor stimulation involve activation of opiate receptors. As naloxone and clonidine do not appear to interact with the same receptor site, the observed functional antagonism suggests the release of an endogenous opiate by clonidine or alpha-methyldopa and the possible role of the opiate in the central control of sympathetic tone."],
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
        "id": "3",
        "offsets": [[0, 8]],
        "text": ["Naloxone"],
        "type": "Chemical",
        "normalized": [{"db_name": "MESH", "db_id": "D009270"}]
    },
    ...
 ],
```

### Events
- Examples: [MLEE](examples/mlee.py)

```
"events": [
    {
        "id": "3",
        "type": "Reaction",
        "trigger": {
            "offsets": [[0,6]],
            "text": ["reacts"]
        },
        "arguments": [
            {
                "role": "theme",
                "ref_id": "5",
            }
            ...
        ],
    }
    ...
],

```

### Coreferences

- Examples: [n2c2 2011: Coreference Challenge](examples/n2c2_2011.py)

```
"coreferences": [
	{
	   "id": "32",
	   "entity_ids": ["1", "10", "23"],
	},
	...
]
```

### Relations
- Examples: [BC5CDR](examples/bc5cdr.py)

```
"relations": [
    {
        "id": "100",
        "type": "chemical-induced disease",
        "arg1_id": "10",
        "arg2_id": "32",
        "normalized": []
    }
]
```

## Question Answering
- [Schema Template](utils/schemas/qa.py)
- Examples: [BioASQ9 Task B](examples/bioasq.py)

```
{
	"id": "0",
	"document_id": "24267510",
	"question_id": "55031181e9bde69634000014",
	"question": "Is RANKL secreted from the cells?",
	"type": "yesno",
	"context": "Osteoprotegerin (OPG) is a soluble secreted factor that acts as a decoy receptor for receptor activator of NF-\u03baB ligand (RANKL)",
	"answer": ["yes"],
}
```

## Textual Entailment

- [Schema Template](utils/schemas/entailment.py)
- Examples: [SciTail](examples/scitail.py)

```
{
	"id": "0",
	"document_id": "NULL",
	"premise": "Pluto rotates once on its axis every 6.39 Earth days;",
	"hypothesis": "Earth rotates on its axis once times in one day.",
	"label": "neutral",
}
```

## Text Pairs

- [Schema Template](utils/schemas/pairs.py)
- Examples: [MQP](examples/mqp.py)

```
{
	"id": "0",
	"document_id": "NULL",
	"text_1": "Am I over weight (192.9) for my age (39)?",
	"text_2": "I am a 39 y/o male currently weighing about 193 lbs. Do you think I am overweight?",
	"label": 1,
}
```


## Text to Text

- [Schema Template](utils/schemas/text_to_text.py)
- Examples: [ParaMed](examples/paramed.py)

```
{
	"id": "0",
	"document_id": "NULL",
	"text_1": "也许 不能 : 分析 结果 提示 激素 疗法 在 维持 去 脂 体重 方面 作用 很小 .",
	"text_2": "probably not : analysis suggests minimal effect of HT in maintaining lean body mass .",
	"text_1_name": "zh",
	"text_2_name": "en",
}
```


## Text
- [Schema Template](utils/schemas/text.py)

```
{
	"id": "0",
	"document_id": "NULL",
	"text": "Am I over weight (192.9) for my age (39)?",
	"label": "question",
}
```
