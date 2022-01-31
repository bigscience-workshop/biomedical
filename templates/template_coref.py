"""
A dataset template for coreference resolution.

{
    "sample_id": "note-1234",
    "text": "Jane Doe loves NLP. It is her favorite thing.",
    "entities": [
        {
            "id": "E0",
            "span_start": 0,
            "span_end": 8,
            "text": "Jane Doe"
        },
        {
            "id": "E1",
            "span_start": 15,
            "span_end": 18,
            "text": "NLP"
        },
        {
            "id": "E2",
            "span_start": 20,
            "span_end": 22,
            "text": "It"
        },
        {
            "id": "E3",
            "span_start": 26,
            "span_end": 29,
            "text": "her"
        }
    ],
    "corefs": [
        {
            "id": "C0",
            "members": [
                "E0",
                "E3"
            ]
        },
        {
            "id": "C1",
            "members": [
                "E1",
                "E2"
            ]
        }
    ]
}


"""

import datasets


features = datasets.Features(
    {
        "sample_id": datasets.Value("string"),
        "text": datasets.Value("string"),
        "entities": datasets.Sequence(
            {
                "id": datasets.Value("string"),
                "span_start": datasets.Value("int32"),
                "span_end": datasets.Value("int32"),
                "text": datasets.Value("string"),
            }
        ),
        "corefs": datasets.Sequence(
            {
                "id": datasets.Value("string"),
                "members": datasets.Sequence(datasets.Value("string")),
            }
        ),
    }
)
