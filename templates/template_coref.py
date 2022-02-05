"""
A dataset template for coreference resolution.

{
    "passages": [
        {
            "document_id": "clinical-103"
            "type": "discharge summary",
            "text": "Jane Doe loves NLP. It is her favorite thing ......",
            "entities": [
                {
                    "entity_id": "clinical-103-0-0-0-2",
                    "offsets": [[0, 8]],
                    "text": "Jane Doe",
                    "type": "person",
                    "entity_kb_id": "",
                },...
            ],
            "coreferences": [
                {
                    "corefernce_id": "clinical-103-0",
                    "entity_ids": ["clinical-103-0-0-0-2", ...],
                },...
        }
    ]
}

"""

import datasets

features = Features(
    {
        "passages": [
            {
                "document_id": Value("string"),
                "type": Value("string"),
                "text": Value("string"),
                "entities": [
                    {
                        "entity_id": Value("string"),
                        "offsets": [[Value("int32")]],
                        "text": Value("string"),
                        "type": Value("string"),
                        "entity_kb_id": Value("string"),
                    }
                ],
                "coreferences": [
                    {
                        "coreference_id": Value("string"),
                        "entity_ids": [Value("string")],
                    }
                ],
            }
        ]
    }
)
