"""
MuchMore Springer Bilingual Corpus

homepage

* https://muchmore.dfki.de/resources1.htm


description of annotation format

* https://muchmore.dfki.de/pubs/D4.1.pdf


"""

from dataclasses import dataclass
import gzip
import os
import re
import tarfile
from typing import Iterable, Tuple
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element

import chardet
import pandas as pd
from datasets import load_dataset

"""
.eng.abstr
.ger.abstr

.eng.abstr.chunkmorph.annotated.xml
.ger.abstr.chunkmorph.annotated.xml
"""

#ds_translation = load_dataset(
#    'muchmore.py',
#    name="translation",
#)

#ds_original = load_dataset(
#    'muchmore.py',
#    name="original",
#)

ds = load_dataset(
    'muchmore.py',
    name="muchmore",
)


for snippet in ds['train'][0]['passages'][0]['snippets']:
    print(snippet['text'])
    start, end = snippet['offsets'][0]
    print(ds['train'][0]['passages'][0]['text'][start:end])
