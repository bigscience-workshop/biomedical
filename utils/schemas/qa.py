"""
Question Answering Schema
"""
import datasets

features = datasets.Features(
    {
        "id": datasets.Value("string"),
        "question_id": datasets.Value("string"),
        "document_id": datasets.Value("string"),
        "question": datasets.Value("string"),
        "type": datasets.Value("string"),
        "context": datasets.Value("string"),
        "answer": datasets.Sequence(datasets.Value("string")),
    }
)
