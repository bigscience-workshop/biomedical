import datasets
import os

import pandas as pd
import numpy as np

from utils import schemas
from utils.configs import BigBioConfig
# from utils.constants import Tasks

_CITATION = """\

}
"""

_DESCRIPTION = """\
We manually annotated two corpora from the biomedical field. The ESSAI corpus contains clinical trial protocols in French. They were mainly obtained from the National Cancer Institute The typical protocol consists of two parts: the summary of the trial, which indicates the purpose of the trial and the methods applied; and a detailed description of the trial with the inclusion and exclusion criteria. The CAS corpus contains clinical cases published in scientific literature and training material. They are published in different journals from French-speaking countries (France, Belgium, Switzerland, Canada, African countries, tropical countries) and are related to various medical specialties (cardiology, urology, oncology, obstetrics, pulmonology, gastro-enterology). The purpose of clinical cases is to describe clinical situations of patients. Hence, their content is close to the content of clinical narratives (description of diagnoses, treatments or procedures, evolution, family history, expected audience, etc.). In clinical cases, the negation is frequently used for describing the patient signs, symptoms, and diagnosis. Speculation is present as well but less frequently.

This version only contain the annotated ESSAI corpus
"""

_HOMEPAGE = "https://clementdalloux.fr/?page_id=28"

_LICENSE = "DUA"

_URLS = {
    "essai_source": "",
    "essai_bigbio_text": "",
}

_SUPPORTED_TASKS = []
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class ESSAI(datasets.GeneratorBasedBuilder):
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    DEFAULT_CONFIG_NAME = "essai_source"

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="essai_source",
            version=SOURCE_VERSION,
            description="ESSAI source schema",
            schema="source",
            subset_id="essai",
        ),
        BigBioConfig(
            name="essai_bigbio_text",
            version=BIGBIO_VERSION,
            description="ESSAI simplified BigBio schema for negation/speculation classification",
            schema="bigbio_text",
            subset_id="essai",
        ),
        BigBioConfig(
            name="essai_bigbio_segm",
            version=BIGBIO_VERSION,
            description="ESSAI simplified BigBio schema for part-of-speech-tagging",
            schema="bigbio_segm",
            subset_id="essai",
        ),
    ]

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                                "id": datasets.Value("string"),
                                "document_id": datasets.Value("string"),
                                "text": [datasets.Value("string")],
                                "lemmas": [datasets.Value("string")],
                                "POS_tags": [datasets.Value("string")],
                                "labels": [datasets.Value("string")],
                }
            )
        elif self.config.schema == "bigbio_text":
            features = schemas.text_features
        elif self.config.schema == "bigbio_segm":
        	features = schemas.segmentation_features
            
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )


    def _split_generators(self, dl_manager):
        urls = _URLS[self.config.name]
        data_dir = "./biodatasets/essai/"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "datadir": data_dir
                },
            ),
        ]

    def _generate_examples(self, datadir):
        key = 0
        for file in ["ESSAI_neg.txt", "ESSAI_spec.txt"]:
            filepath = os.path.join(datadir, file)
            label = "negation" if "neg" in file else "speculation"
            id_docs = []
            id_words = []
            words = []
            lemmas = []
            POS_tags = []

            with open(filepath) as f:
              for line in f.readlines():
                line_content = line.split("\t")
                if len(line_content) > 1 :
                  id_docs.append(line_content[0])
                  id_words.append(line_content[1])
                  words.append(line_content[2])
                  lemmas.append(line_content[3])
                  POS_tags.append(line_content[4])

            dic = {
                "id_docs": np.array(list(map(int, id_docs))),
                "id_words": id_words,
                "words": words,
                "lemmas": lemmas,
                "POS_tags": POS_tags,
            }
            if self.config.schema == "source":
                for doc_id in set(dic["id_docs"]):
                    idces = np.argwhere(dic["id_docs"]==doc_id)[:,0]
                    text = [dic["words"][id] for id in idces]
                    text_lemmas = [dic["lemmas"][id] for id in idces]
                    POS_tags_ = [dic["POS_tags"][id] for id in idces]
                    yield key, {
                                "id": key,
                                "document_id": doc_id,
                                "text": text,
                                "lemmas": text_lemmas,
                                "POS_tags": POS_tags_,
                                "labels": [label],
                                }
                    key += 1
            elif self.config.schema == "bigbio_text":
                for doc_id in set(dic["id_docs"]):
                    idces = np.argwhere(dic["id_docs"]==doc_id)[:,0]
                    text = (" ".join([dic["words"][id] for id in idces]))
                    yield key, {
                                "id": key,
                                "document_id": doc_id,
                                "text": text,
                                "labels": [label],
                                }
                    key += 1
            elif self.config.schema == "bigbio_segm":
                for doc_id in set(dic["id_docs"]):
                    idces = np.argwhere(dic["id_docs"]==doc_id)[:,0]
                    text = [dic["words"][id] for id in idces]
                    POS_tags_ = [dic["POS_tags"][id] for id in idces]
                    yield key, {
                                "id": key,
                                "document_id": doc_id,
                                "text": text,
                                "labels": POS_tags_,
                                }
                    key += 1
                    