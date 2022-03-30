"""
General Text Schema
"""
import datasets


features = datasets.Features(
    {
        "id": datasets.Value("string"),
        "document_id": datasets.Value("string"),
        "text": datasets.Value("string"),
        "label": datasets.Value("string"),
    }
)
