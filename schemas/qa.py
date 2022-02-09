"""
Question Answering Schema
"""
import datasets


features = datasets.Features(
    {
        "id": datasets.Value("string"),
        "question": datasets.Value("string"),
        "type": datasets.Value("string"),
        "context": datasets.Value("string"),
        "document": {
            "document_id": datasets.Value("string"),
            "text": datasets.Value("string"),
        },
        "answer": datasets.Sequence(datasets.Value("string")),
        "expanded_answer": datasets.Sequence(datasets.Value("string")),
    }
)
