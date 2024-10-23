# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
We present the SourceData-NLP dataset produced through the routine curation of papers during the publication process.
A unique feature of this dataset is its emphasis on the annotation of bioentities in figure legends.
We annotate eight classes of biomedical entities (small molecules, gene products, subcellular components,
cell lines, cell types, tissues, organisms, and diseases), their role in the experimental design,
and the nature of the experimental method as an additional class.
SourceData-NLP contains more than 620,000 annotated biomedical entities, curated from 18,689 figures in
3,223 papers in molecular and cell biology.

[bigbio_schema_name] = kb
"""

import itertools
import json
import os
from typing import Dict, List, Tuple

import datasets

from .bigbiohub import BigBioConfig, Tasks, kb_features

_LANGUAGES = ["English"]
_PUBMED = True
_LOCAL = False
_DISPLAYNAME = "SourceData-NLP"

_CITATION = """\
@article{abreu2023sourcedata,
  title={The SourceData-NLP dataset: integrating curation into scientific publishing
  for training large language models},
  author={Abreu-Vicente, Jorge and Sonntag, Hannah and Eidens, Thomas and Lemberger, Thomas},
  journal={arXiv preprint arXiv:2310.20440},
  year={2023}
}
"""

_DATASETNAME = "sourcedata_nlp"

_DESCRIPTION = """\
SourceData is an NER/NED dataset of expert annotations of nine
entity types in figure captions from biomedical research papers.
"""

_HOMEPAGE = "https://sourcedata.embo.org/"


_LICENSE = "CC_BY_4p0"


_URLS = {
    _DATASETNAME: (
        "https://huggingface.co/datasets/EMBO/SourceData/resolve/main/bigbio/source_data_json_splits_2.0.2.zip"
    )
}


_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_DISAMBIGUATION, Tasks.NAMED_ENTITY_RECOGNITION]

_SOURCE_VERSION = "2.0.2"

_BIGBIO_VERSION = "1.0.0"


class SourceDataNlpDataset(datasets.GeneratorBasedBuilder):
    """NER + NED dataset of multiple entity types from figure captions of scientific publications"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="sourcedata_nlp_source",
            version=SOURCE_VERSION,
            description="sourcedata_nlp source schema",
            schema="source",
            subset_id="sourcedata_nlp",
        ),
        BigBioConfig(
            name="sourcedata_nlp_bigbio_kb",
            version=BIGBIO_VERSION,
            description="sourcedata_nlp BigBio schema",
            schema="bigbio_kb",
            subset_id="sourcedata_nlp",
        ),
    ]

    DEFAULT_CONFIG_NAME = "sourcedata_nlp_source"

    def _info(self) -> datasets.DatasetInfo:
        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "doi": datasets.Value("string"),
                    "abstract": datasets.Value("string"),
                    "figures": [
                        {
                            "fig_id": datasets.Value("string"),
                            "label": datasets.Value("string"),
                            "fig_graphic_url": datasets.Value("string"),
                            "panels": [
                                {
                                    "panel_id": datasets.Value("string"),
                                    "text": datasets.Value("string"),
                                    "panel_graphic_url": datasets.Value("string"),
                                    "entities": [
                                        {
                                            "annotation_id": datasets.Value("string"),
                                            "source": datasets.Value("string"),
                                            "category": datasets.Value("string"),
                                            "entity_type": datasets.Value("string"),
                                            "role": datasets.Value("string"),
                                            "text": datasets.Value("string"),
                                            "ext_ids": datasets.Value("string"),
                                            "norm_text": datasets.Value("string"),
                                            "ext_dbs": datasets.Value("string"),
                                            "in_caption": datasets.Value("bool"),
                                            "ext_names": datasets.Value("string"),
                                            "ext_tax_ids": datasets.Value("string"),
                                            "ext_tax_names": datasets.Value("string"),
                                            "ext_urls": datasets.Value("string"),
                                            "offsets": [datasets.Value("int64")],
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                }
            )

        elif self.config.schema == "bigbio_kb":
            features = kb_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""

        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train.jsonl"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "test.jsonl"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "validation.jsonl"),
                },
            ),
        ]

    def _generate_examples(self, filepath) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        if self.config.schema == "source":
            with open(filepath) as fstream:
                for line in fstream:
                    document = self._parse_document(line)
                    doc_figs = document["figures"]
                    all_figures = []
                    for fig in doc_figs:
                        all_panels = []
                        figure = {
                            "fig_id": fig["fig_id"],
                            "label": fig["label"],
                            "fig_graphic_url": fig["fig_graphic_url"],
                        }
                        for p in fig["panels"]:
                            panel = {
                                "panel_id": p["panel_id"],
                                "text": p["text"].strip(),
                                "panel_graphic_url": p["panel_graphic_url"],
                                "entities": [
                                    {
                                        "annotation_id": t["tag_id"],
                                        "source": t["source"],
                                        "category": t["category"],
                                        "entity_type": t["entity_type"],
                                        "role": t["role"],
                                        "text": t["text"],
                                        "ext_ids": t["ext_ids"],
                                        "norm_text": t["norm_text"],
                                        "ext_dbs": t["ext_dbs"],
                                        "in_caption": bool(t["in_caption"]),
                                        "ext_names": t["ext_names"],
                                        "ext_tax_ids": t["ext_tax_ids"],
                                        "ext_tax_names": t["ext_tax_names"],
                                        "ext_urls": t["ext_urls"],
                                        "offsets": t["local_offsets"],
                                    }
                                    for t in p["tags"]
                                ],
                            }
                            for e in panel["entities"]:
                                assert type(e["offsets"]) == list
                            if len(panel["entities"]) == 0:
                                continue
                            all_panels.append(panel)

                        figure["panels"] = all_panels

                        # Pass on all figures that aren't split into panels
                        if len(all_panels) == 0:
                            continue
                        all_figures.append(figure)

                    output = {
                        "doi": document["doi"],
                        "abstract": document["abstract"],
                        "figures": all_figures,
                    }
                    yield document["doi"], output

        elif self.config.schema == "bigbio_kb":
            uid = itertools.count(0)

            with open(filepath) as fstream:
                for line in fstream:
                    output = {}
                    document = self._parse_document(line)

                    # Get ids for each document + list of passages
                    output["id"] = next(uid)
                    output["document_id"] = document["doi"]
                    output["passages"] = document["passages"]
                    for i, passage in enumerate(output["passages"]):
                        passage["id"] = next(uid)
                        passage_text = passage["text"].strip()
                        passage["text"] = [passage_text]
                        passage_offsets = passage["offsets"]
                        if i == 0:
                            passage_offsets[1] = len(passage_text.strip())
                        passage["offsets"] = [
                            [
                                passage_offsets[0],
                                passage_offsets[0] + passage_offsets[1],
                            ]
                        ]
                    entities = []
                    for fig in document["figures"]:
                        for panel in fig["panels"]:
                            for tag in panel["tags"]:
                                # Create two separate ents if both role and tag are labeled.
                                ent_type = self._get_entity_type(tag)
                                if ent_type is not None:
                                    ent = {
                                        "id": next(uid),
                                        "type": ent_type,
                                        "text": [tag["text"]],
                                        "offsets": [tag["document_offsets"]],
                                        "normalized": [
                                            {"db_name": db_name, "db_id": db_id}
                                            for db_name, db_id in zip(tag["ext_dbs"], tag["ext_ids"])
                                        ],
                                    }
                                    entities.append(ent)

                                # When entity has a role as well, add an additional entity for this
                                # Necessary to create duplicate entity due to constraints of BigBio schema
                                # These can be consolidated by matching up document ID + offsets
                                role = self._get_entity_role(tag)
                                if role is not None:
                                    role_ent = {
                                        "id": next(uid),
                                        "type": role,
                                        "text": [tag["text"]],
                                        "offsets": [tag["document_offsets"]],
                                        "normalized": [
                                            {"db_name": db_name, "db_id": db_id}
                                            for db_name, db_id in zip(tag["ext_dbs"], tag["ext_ids"])
                                        ],
                                    }
                                    entities.append(role_ent)

                    output["entities"] = entities

                    output["relations"] = []
                    output["events"] = []
                    output["coreferences"] = []

                    yield output["document_id"], output

    def _parse_document(self, raw_document):
        doc = json.loads(raw_document.strip())
        return doc

    def _get_entity_type(self, tag):
        if tag["entity_type"] == "molecule":
            return "SMALL_MOLECULE"
        elif tag["entity_type"] in ["geneprod", "gene", "protein"]:
            return "GENEPROD"
        elif tag["entity_type"] == "subcellular":
            return "SUBCELLULAR"
        elif tag["entity_type"] == "cell_type":
            return "CELL_TYPE"
        elif tag["entity_type"] == "tissue":
            return "TISSUE"
        elif tag["entity_type"] == "organism":
            return "ORGANISM"
        elif tag["category"] == "assay":
            return "EXP_ASSAY"
        elif tag["category"] == "disease":
            return "DISEASE"
        elif tag["entity_type"] == "cell_line":
            return "CELL_LINE"

    def _get_entity_role(self, tag):
        if tag["entity_type"] == "molecule":
            if tag["role"] == "intervention":
                return "CONTROLLED_VAR"
            elif tag["role"] == "assayed":
                return "MEASURED_VAR"
        elif tag["entity_type"] in ["geneprod", "gene", "protein"]:
            if tag["role"] == "intervention":
                return "CONTROLLED_VAR"
            elif tag["role"] == "assayed":
                return "MEASURED_VAR"
