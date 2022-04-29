""" 
Text to Text Schema

Several tasks boil down to transforming 1 string into annother string, including:

- Translation
- Summarization
- Paraphrasing 
"""
import datasets

features = datasets.Features(
    {
        "id": datasets.Value("string"),
        "document_id": datasets.Value("string"),
        "text_1": datasets.Value("string"),
        "text_2": datasets.Value("string"),
        "text_1_name": datasets.Value("string"),
        "text_2_name": datasets.Value("string"),
    }
)
