"""
Textual Entailment Schema
"""
import datasets

features = datasets.Features(
    {
        "id": datasets.Value("string"),
        "premise": datasets.Value("string"),
        "hypothesis": datasets.Value("string"),
        "label": datasets.Value("string"),
    }
)
