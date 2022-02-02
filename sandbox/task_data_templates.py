import datasets
from datasets import ClassLabel, Features, Sequence, Value


# ag news style + metadata
# https://huggingface.co/datasets/ag_news
text_classification_single_v0 = Features(
    {
        "text": Value("string"),
        "label": ClassLabel(names=["World", "Sports", "Business", "Sci/Tech"]),
        "metadata": {
            "anyvar_youwant": Value("int32"),
        }
    }
)




# lanuage codes, strings, and optional metadata
translation_v0 = Features(
    {
        "en": Value("string"),
        "zh": Value("string"),
        # allow for arbitrary metadata
        "metadata": {
            "sample_id_prefix": Value("string"),
            "sample_id_en": Value("string"),
            "sample_id_de": Value("string"),
        },
    }
)



ner_rel_v0 = Features(
    {
        "article_id": Value("int32"),
        "text": Value("string"),
        "entities": Sequence(
            {
                "spans": Sequence(
                    Value("int32")
                ),
                "text": Value("string"),
                "entity_type": Value("string"),
            }
        ),
        "relations": Sequence(
            {
                "relation_type": Value("string"),
                "arg1": Value("string"),
                "arg2": Value("string"),
            }
        ),
    }
)



general_v0 = Features(
    {
        "passages": Sequence(
            {
                "document_id": Value("string"),
                "type": Value("string"),
                "text": Value("string"),

                "snippets": Sequence(
                    {
                        "snippet_id": Value("string"),
                        "offsets": Sequence([Value("int32")]),
                        "text": Value("string"),
                        "type": Value("string"),

                    }
                ),
                "entities": Sequence(
                    {
                        "entity_id": Value("int32"),
                        "offsets": Sequence([Value("int32")]),
                        "text": Value("string"),
                        "type": Value("string"),
                        "entity_kb_id": Value("string"),
                    }
                ),
                "relations": Sequence(
                    {
                        'relation_id': Value("string"),
                        'type': Value("string"),
                        'arg1_id': Value("int32"),
                        'arg2_id': Value("int32"),
                        "relation_kb_id": Value("int32"),
                    }
                )
            }
        )
    }
)



concept_v0 = Features(
    {
        "concepts": datasets.Sequence(
            {
                "concept_id": Value("string"),   # C1
                "class": Value("string"),        # "entity"
                "offsets": Sequence([Value("int32")]),   # [(100,109), (120, 123)]
                "text": Sequence([Value("string")]),     # [["Arthritis"], ["Rheumatoid"]]
                "type": Value("string"),                 # "disease"
                "corefs": Sequence(Value("int32")),      # [13, 20],
                "concept_kb_id": Sequence(
                    {
                        'kb_name': Value("string"),    # MESH
                        'kb_id': Value("string"),      # 'D001172',
                    },
                ),
            }
        )
    }
)
