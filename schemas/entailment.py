"""
Knowledge Base Schema
"""
import datasets


features = datasets.Features(
    {
        "question": datasets.Value("string"),
        "text1": datasets.Value("string"),  # Passage/sentence 1
        "text2": datasets.Value("string"),  # Passage/sentence
        "label": datasets.Value("string"),  # entails/etc
    }
)
