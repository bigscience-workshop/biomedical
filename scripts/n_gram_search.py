# partially from https://gist.github.com/gaulinmp/da5825de975ed0ea6a24186434c24fe4
from nltk.util import ngrams
from nltk.corpus import stopwords
import spacy
import pandas as pd
import re
from itertools import chain
from collections import Counter
from datasets import load_dataset


nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("sentencizer")
STOPWORDS = nlp.Defaults.stop_words

# N = 5
re_sent_ends_naive = re.compile(r'[.\n]')
re_stripper_naive = re.compile('[^a-zA-Z\.\n]')

splitter_naive = lambda x: re_sent_ends_naive.split(re_stripper_naive.sub(' ', x))


# list of tokens for one sentence
def remove_stop_words(text):
    result = []
    for w in text:
        if w not in STOPWORDS:
            result.append(w)
    return result


# get sentence from multiple sentences
def parse_sentences(text, nlp):
    doc = nlp(text)
    sentences = (remove_stop_words(sent) for sent in doc.sents)
    return sentences


def get_tuples_manual_sentences(txt, N):
    """Naive get tuples that uses periods or newlines to denote sentences."""
    if not txt:
        return None, []
    sentences = (x.split() for x in splitter_naive(txt) if x)
    sentences = list(map(remove_stop_words, list(sentences)))
    # sentences = (remove_stop_words(nlp(x)) for x in splitter_naive(txt) if x)
    # sentences = parse_sentences(txt, nlp)
    # print(list(sentences))
    ng = (ngrams(x, N) for x in sentences if len(x) >= N)
    return sentences, list(chain(*ng))


def count_by_split(split_data):
    c = Counter()
    for entry in split_data:
        text = entry['text']
        sents, tup = get_tuples_manual_sentences(text, N)
        tup = ["_".join(ta) for ta in tup]
        c.update(tup)
    return c


# data = load_dataset("bigbio/biodatasets/chemdner/chemdner.py", name="chemdner_bigbio_text")
# counters = []
# for split, split_data in data.items():
#     split_counter = count_by_split(split_data)
#     counters.append(split_counter)

# ab_intersect = counters[0] & counters[1]
# diff = {x: count for x, count in counters[0].items() if x not in ab_intersect.keys() and count > 2}
# if len(counters) > 2:
#     bc_intersect = counters[1] & counters[2]
# print(ab_intersect.most_common(10))
# print(Counter(diff).most_common(10))
# data.cleanup_cache_files()
