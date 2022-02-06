"""
MuchMore Springer Bilingual Corpus

homepage

* https://muchmore.dfki.de/resources1.htm


description of annotation format

* https://muchmore.dfki.de/pubs/D4.1.pdf


"""
from datasets import load_dataset

"""
.eng.abstr
.ger.abstr

.eng.abstr.chunkmorph.annotated.xml
.ger.abstr.chunkmorph.annotated.xml
"""

ds_original = load_dataset(
    'muchmore.py',
    name="original",
)

ds = load_dataset(
    'muchmore.py',
    name="muchmore",
)

ds_en = load_dataset(
    'muchmore.py',
    name="muchmore_en",
)

ds_de = load_dataset(
    'muchmore.py',
    name="muchmore_de",
)

ds_plain = load_dataset(
    'muchmore.py',
    name="plain",
)

ds_plain_en = load_dataset(
    'muchmore.py',
    name="plain_en",
)

ds_plain_de = load_dataset(
    'muchmore.py',
    name="plain_de",
)

ds_translation = load_dataset(
    'muchmore.py',
    name="translation",
)


for sample in ds['train']:

    # assert snippets (sentences) line up with main text
    for snippet in sample['passages'][0]['snippets']:
        snip_text = snippet['text']
        start, end = snippet['offsets']
        main_text = sample['passages'][0]['text'][start:end]
        assert(snip_text == main_text)

    # assert entities line up with main text
    for entity in sample['passages'][0]['entities']:
        entity_text = entity['text'][0]
        start, end = entity['offsets'][0]
        main_text = sample['passages'][0]['text'][start:end]
        assert(entity_text == main_text)
