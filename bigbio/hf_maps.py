"""
Functions to be used with the datasets map function
* https://huggingface.co/docs/datasets/main/en/nlp_process#map
"""

def text_from_kb(passages):
    return " ".join([t for p in passages for t in p["text"]])

def map_text_from_kb(example):
    return {"text": text_from_kb(example["passages"])}

def map_batch_text_from_kb(examples):
    return {"text": [text_from_kb(passages) for passages in examples["passages"]]}


def text_from_te(premise, hypothesis):
    return premise + " " + hypothesis

def map_text_from_te(example):
    return {"text": text_from_te(example["premise"], example["hypothesis"])}

def map_batch_text_from_te(examples):
    return {
        "text": [
            text_from_te(premise, hypothesis)
            for premise, hypothesis in zip(examples["premise"], examples["hypothesis"])
        ]
    }


def text_from_pairs(text_1, text_2):
    return text_1 + " " + text_2

def map_text_from_pairs(example):
    return {"text": text_from_pairs(example["text_1"], example["text_2"])}

def map_batch_text_from_pairs(examples):
    return {
        "text": [
            text_from_pairs(text_1, text_2)
            for text_1, text_2 in zip(examples["text_1"], examples["text_2"])
        ]
    }


def text_from_t2t(text_1, text_2):
    return text_1

def map_text_from_t2t(example):
    return {"text": text_from_t2t(example["text_1"], example["text_2"])}

def map_batch_text_from_t2t(examples):
    return {
        "text": [
            text_from_t2t(text_1, text_2)
            for text_1, text_2 in zip(examples["text_1"], examples["text_2"])
        ]
    }


def text_from_text(text):
    return text

def map_text_from_text(example):
    return {"text": text_from_text(example["text"])}

def map_batch_text_from_text(examples):
    return {"text": [text_from_text(text) for text in examples["text"]]}



def text_from_qa(question, qtype, choices, context, answer):
    return "{} {} {} {} {}".format(
        question,
        qtype,
        " ".join(choices),
        context,
        " ".join(answer),
    )

def map_text_from_qa(example):
    return {"text": text_from_qa(
        example["question"],
        example["type"],
        example["choices"],
        example["context"],
        example["answer"],
    )}

def map_batch_text_from_qa(examples):
    return {
        "text": [
            text_from_qa(question, qtype, choices, context, answer)
            for question, qtype, choices, context, answer in zip(
                    examples["question"],
                    examples["type"],
                    examples["choices"],
                    examples["context"],
                    examples["answer"],
             )
        ]
    }


BATCH_MAPPERS_TEXT_FROM_SCHEMA = {
    "kb": map_batch_text_from_kb,
    "te": map_batch_text_from_te,
    "pairs": map_batch_text_from_pairs,
    "text": map_batch_text_from_text,
    "qa": map_batch_text_from_qa,
    "t2t": map_batch_text_from_t2t,
}
