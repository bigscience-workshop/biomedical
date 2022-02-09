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
        "document_id": datasets.Value("string"),
        "passages": datasets.Sequence(
            {
                "id": datasets.Value("int32"),
                "type": datasets.Value("string"),
                "text": datasets.Value("string"),
                "offsets": datasets.Sequence([datasets.Value("int32")]),
            }
        ),
        "entities": datasets.Sequence(
            {
                "id": datasets.Value("int32"),
                "offsets": datasets.Sequence([datasets.Value("int32")]),
                "text": datasets.Sequence([datasets.Value("string")]),
                "type": datasets.Value("string"),
                "normalized": datasets.Sequence(
                    {
                        "db_name": datasets.Value("string"),
                        "db_id": datasets.Value("string"),
                    }
                ),
            }
        ),
        "events": datasets.Sequence(
            {
                "id": datasets.Value("int32"),
                "offsets": datasets.Sequence([datasets.Value("int32")]),
                "text": datasets.Sequence([datasets.Value("string")]),
                "type": datasets.Value("string"),
                "normalized": datasets.Sequence(
                    {
                        "db_name": datasets.Value("string"),
                        "db_id": datasets.Value("string"),
                    }
                ),
            }
        ),
        "coreferences": datasets.Sequence(
            {
                "id": datasets.Value("int32"),
                "entity_ids": [datasets.Value("string")],
            }
        ),
        "relations": datasets.Sequence(
            {
                "id": datasets.Value("string"),
                "type": datasets.Value("string"),
                "arg1_id": datasets.Value("int32"),
                "arg2_id": datasets.Value("int32"),
                "normalized": datasets.Sequence(
                    {
                        "db_name": datasets.Value("string"),
                        "db_id": datasets.Value("string"),
                    }
                ),
            }
        ),
    }
)


# foobar = {
#     "entities": [
#         {
#             "id": 2,
#             "offsets": [[0, 8]],
#             "text": "Naloxone",
#             "type": "Chemical",
#             "normalized": [
#                 {
#                     "db_name": "MESH",
#                     "db_id": "D009270",
#                 }
#             ],
#         },
#         {
#             "id": 3,
#             "offsets": [[49, 58]],
#             "text": "clonidine",
#             "type": "Chemical",
#             "normalized": [
#                 {
#                     "db_name": "MESH",
#                     "db_id": "D003000",
#                 }
#             ],
#         },
#     ],
# }
