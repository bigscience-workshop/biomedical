"""
Knowledge Base Schema

This is a very general schema that covers many information extraction-style
tasks, including:

- Named entity recognition (NER)
- Named entity disambiguation/canonicalization/normalization (NED)
- Event extraction
- Relation extraction (RE)
- Coreference resolution

This schema assumes a document with child elements (e.g., entities, relations)
and a flat hierarchy of document passages.

"""
import datasets


features = datasets.Features(
    {
        "id": datasets.Value("string"),
        "document_id": datasets.Value("string"),
        "passages": [
            {
                "id": datasets.Value("string"),
                "type": datasets.Value("string"),
                "text": datasets.Value("string"),
                "offsets": datasets.Sequence(datasets.Value("int32")),
            }
        ],
        "entities": [
            {
                "id": datasets.Value("string"),
                "offsets": datasets.Sequence([datasets.Value("int32")]),
                "text": datasets.Sequence(datasets.Value("string")),
                "type": datasets.Value("string"),
                "normalized": [
                    {
                        "db_name": datasets.Value("string"),
                        "db_id": datasets.Value("string"),
                    }
                ],
            }
        ],
        "events": [
            {
                "id": datasets.Value("string"),
                "type": datasets.Value("string"),
                # refers to the text_bound_annotation of the trigger
                "trigger": {
                    "offsets": datasets.Sequence([datasets.Value("int32")]),
                    "text": datasets.Value("string")
                },
                "arguments": [
                    {
                        "role": datasets.Value("string"),
                        "ref_id": datasets.Value("string"),
                    }
                ],
            }
        ],
        "coreferences": [
            {
                "id": datasets.Value("string"),
                "entity_ids": datasets.Sequence(datasets.Value("string")),
            }
        ],
        "relations": [
            {
                "id": datasets.Value("string"),
                "type": datasets.Value("string"),
                "arg1_id": datasets.Value("string"),
                "arg2_id": datasets.Value("string"),
                "normalized": [
                    {
                        "db_name": datasets.Value("string"),
                        "db_id": datasets.Value("string"),
                    }
                ],
            }
        ],
    }
)
