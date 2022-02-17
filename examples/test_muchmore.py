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

ds_source = load_dataset(
    'muchmore.py',
    name="source",
)

ds_kb = load_dataset(
    'muchmore.py',
    name="bigbio-kb",
)

ds_kb_en = load_dataset(
    "muchmore.py",
    name="bigbio-kb-en",
)

ds_kb_de = load_dataset(
    'muchmore.py',
    name="bigbio-kb-de",
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
    "muchmore.py",
    name="bigbio-translation",
)


for sample in ds_kb_en["train"]:

    # assert entities line up with main text
    for entity in sample["entities"]:
        entity_text = entity["text"][0]
        start, end = entity["offsets"][0]
        main_text = sample["passages"][0]["text"][0][start:end]
        assert entity_text == main_text
