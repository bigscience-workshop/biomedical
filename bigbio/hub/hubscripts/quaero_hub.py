import re
from pathlib import Path

import datasets

from .bigbiohub import kb_features
from .bigbiohub import BigBioConfig
from .bigbiohub import Tasks

_LANGUAGES = ['French']
_PUBMED = True
_LOCAL = False
_CITATION = """\
@InProceedings{neveol14quaero, 
  author = {Névéol, Aurélie and Grouin, Cyril and Leixa, Jeremy 
            and Rosset, Sophie and Zweigenbaum, Pierre},
  title = {The {QUAERO} {French} Medical Corpus: A Ressource for
           Medical Entity Recognition and Normalization}, 
  OPTbooktitle = {Proceedings of the Fourth Workshop on Building 
                 and Evaluating Ressources for Health and Biomedical 
                 Text Processing}, 
  booktitle = {Proc of BioTextMining Work}, 
  OPTseries = {BioTxtM 2014}, 
  year = {2014}, 
  pages = {24--30}, 
}
"""

_DESCRIPTION = """\
The QUAERO French Medical Corpus has been initially developed as a resource for named entity recognition and normalization [1]. It was then improved with the purpose of creating a gold standard set of normalized entities for French biomedical text, that was used in the CLEF eHealth evaluation lab [2][3].

A selection of MEDLINE titles and EMEA documents were manually annotated. The annotation process was guided by concepts in the Unified Medical Language System (UMLS):

1. Ten types of clinical entities, as defined by the following UMLS Semantic Groups (Bodenreider and McCray 2003) were annotated: Anatomy, Chemical and Drugs, Devices, Disorders, Geographic Areas, Living Beings, Objects, Phenomena, Physiology, Procedures.

2. The annotations were made in a comprehensive fashion, so that nested entities were marked, and entities could be mapped to more than one UMLS concept. In particular: (a) If a mention can refer to more than one Semantic Group, all the relevant Semantic Groups should be annotated. For instance, the mention “récidive” (recurrence) in the phrase “prévention des récidives” (recurrence prevention) should be annotated with the category “DISORDER” (CUI C2825055) and the category “PHENOMENON” (CUI C0034897); (b) If a mention can refer to more than one UMLS concept within the same Semantic Group, all the relevant concepts should be annotated. For instance, the mention “maniaques” (obsessive) in the phrase “patients maniaques” (obsessive patients) should be annotated with CUIs C0564408 and C0338831 (category “DISORDER”); (c) Entities which span overlaps with that of another entity should still be annotated. For instance, in the phrase “infarctus du myocarde” (myocardial infarction), the mention “myocarde” (myocardium) should be annotated with category “ANATOMY” (CUI C0027061) and the mention “infarctus du myocarde” should be annotated with category “DISORDER” (CUI C0027051)

The QUAERO French Medical Corpus BioC release comprises a subset of the QUAERO French Medical corpus, as follows:

Training data (BRAT version used in CLEF eHealth 2015 task 1b as training data): 
- MEDLINE_train_bioc file: 833 MEDLINE titles, annotated with normalized entities in the BioC format 
- EMEA_train_bioc file: 3 EMEA documents, segmented into 11 sub-documents, annotated with normalized entities in the BioC format 

Development data  (BRAT version used in CLEF eHealth 2015 task 1b as test data and in CLEF eHealth 2016 task 2 as development data): 
- MEDLINE_dev_bioc file: 832 MEDLINE titles, annotated with normalized entities in the BioC format
- EMEA_dev_bioc file: 3 EMEA documents, segmented into 12 sub-documents, annotated with normalized entities in the BioC format 

Test data (BRAT version used in CLEF eHealth 2016 task 2 as test data): 
- MEDLINE_test_bioc folder: 833 MEDLINE titles, annotated with normalized entities in the BioC format 
- EMEA folder_test_bioc: 4 EMEA documents, segmented into 15 sub-documents, annotated with normalized entities in the BioC format 



This release of the QUAERO French medical corpus, BioC version, comes in the BioC format, through automatic conversion from the original BRAT format obtained with the Brat2BioC tool https://bitbucket.org/nicta_biomed/brat2bioc developped by Jimeno Yepes et al.

Antonio Jimeno Yepes, Mariana Neves, Karin Verspoor 
Brat2BioC: conversion tool between brat and BioC
BioCreative IV track 1 - BioC: The BioCreative Interoperability Initiative, 2013


Please note that the original version of the QUAERO corpus distributed in the CLEF eHealth challenge 2015 and 2016 came in the BRAT stand alone format. It was distributed with the CLEF eHealth evaluation tool. This original distribution of the QUAERO French Medical corpus is available separately from https://quaerofrenchmed.limsi.fr  

All questions regarding the task or data should be addressed to aurelie.neveol@limsi.fr
"""

_HOMEPAGE = "https://quaerofrenchmed.limsi.fr/"

_LICENSE = 'GNU Free Documentation License v1.3'

_URL = "https://quaerofrenchmed.limsi.fr/QUAERO_FrenchMed_brat.zip"

_DATASET_NAME = "quaero"
_DISPLAYNAME = "QUAERO"

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.NAMED_ENTITY_DISAMBIGUATION]
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class QUAERO(datasets.GeneratorBasedBuilder):

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="quaero_emea_source",
            version=SOURCE_VERSION,
            description="QUAERO source schema on the EMEA subset",
            schema="source",
            subset_id="quaero_emea",
        ),
        BigBioConfig(
            name="quaero_medline_source",
            version=SOURCE_VERSION,
            description="QUAERO source schema on the MEDLINE subset",
            schema="source",
            subset_id="quaero_medline",
        ),
        BigBioConfig(
            name="quaero_emea_bigbio_kb",
            version=BIGBIO_VERSION,
            description="QUAERO simplified BigBio schema on the EMEA subset",
            schema="bigbio_kb",
            subset_id="quaero_emea",
        ),
        BigBioConfig(
            name="quaero_medline_bigbio_kb",
            version=BIGBIO_VERSION,
            description="QUAERO simplified BigBio schema on the MEDLINE subset",
            schema="bigbio_kb",
            subset_id="quaero_medline",
        ),
    ]

    def _info(self):
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document_id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "text_bound_annotations": [  # T line in brat, e.g. type or event trigger
                        {
                            "offsets": datasets.Sequence([datasets.Value("int32")]),
                            "text": datasets.Sequence(datasets.Value("string")),
                            "type": datasets.Value("string"),
                            "id": datasets.Value("string"),
                        }
                    ],
                    "notes": [  # # lines in brat
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "ref_id": datasets.Value("string"),
                            "text": datasets.Value("string"),
                        }
                    ],
                }
            )
        elif self.config.schema == "bigbio_kb":
            features = kb_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=str(_LICENSE),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        urls = _URL
        data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": data_dir, "split": "test"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir,
                    "split": "dev",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        if self.config.subset_id == "quaero_emea":
            subset = "EMEA"
        elif self.config.subset_id == "quaero_medline":
            subset = "MEDLINE"

        folder = Path(filepath) / "QUAERO_FrenchMed" / "corpus" / split / subset

        if self.config.schema == "source":
            for guid, txt_file in enumerate(sorted(folder.glob("*.txt"))):
                example = parse_brat_file(txt_file, parse_notes=True)
                example["id"] = guid
                # Remove unused items from BRAT
                del example["events"]
                del example["relations"]
                del example["equivalences"]
                del example["attributes"]
                del example["normalizations"]
                yield guid, example
        elif self.config.schema == "bigbio_kb":
            for guid, txt_file in enumerate(sorted(folder.glob("*.txt"))):
                example = parse_brat_file(txt_file, parse_notes=True)
                annotator_notes = example["notes"]
                document_id = example["document_id"]

                example = brat_parse_to_bigbio_kb(example)
                example["id"] = guid

                for note in annotator_notes:
                    entity_id = f'{document_id}_{note["ref_id"]}'
                    for e in example["entities"]:
                        if e["id"] == entity_id:
                            for cui in re.split("[\s\+,]", note["text"].strip()):
                                if cui:
                                    e["normalized"].append({"db_id": cui, "db_name": "UMLS"})
                yield guid, example
        else:
            raise ValueError(f"Invalid config: {self.config.name}")
