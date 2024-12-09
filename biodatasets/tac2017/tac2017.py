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
Drug labels (prescribing information or package inserts) describe what a particular medicine
is supposed to do, who should or should not take it, how to use it, and specific safety concerns. 
The US Food and Drug Administration (FDA) publishes regulations governing the content and format 
of this information to provide recommendations for applicants developing labeling for new drugs 
and revising labeling for already approved drugs. One of the major aspects of drug information 
are safety concerns in the form of Adverse Drug Reactions (ADRs). In this evaluation, we are 
focusing on extraction of ADRs from the prescription drug labels.
"""

import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

import datasets

from utils import schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks

_CITATION = """\
@article{,
  author    = {Kirk Roberts and
                Dina Demner-Fushman and
                Joseph M. Tonning},
  title     = {Overview of the TAC 2017 Adverse Reaction Extraction from Drug Labels Track},
  journal   = {Proceedings of the Text Analysis Conference (TAC) 2017, November 13-14 2017, Gaithersburg MD USA},
  volume    = {},
  year      = {2017},
  url       = {https://tac.nist.gov/publications/2017/additional.papers/TAC2017.ADR_overview.proceedings.pdf},
  doi       = {},
  biburl    = {},
  bibsource = {}
}
"""
_DATASETNAME = "tac2017"

_DESCRIPTION = """\
This dataset is designed for extraction of ADRs from prescription drug labels.
"""
_HOMEPAGE = "https://bionlp.nlm.nih.gov/tac2017adversereactions/"

_LICENSE = "None provided."

_URLS = {
    "tac2017": "https://bionlp.nlm.nih.gov/tac2017adversereactions/train_xml.tar.gz",
}

_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_DISAMBIGUATION, Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]

_SOURCE_VERSION = "1.0.0"

_BIGBIO_VERSION = "1.0.0"


class Tac2017Dataset(datasets.GeneratorBasedBuilder):
    """The TAC 2017 dataset is designed for extraction of ADRs from prescription drug labels."""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="tac2017_source",
            version=SOURCE_VERSION,
            description="TAC 2017 source schema",
            schema="source",
            subset_id="tac2017",
        ),
        BigBioConfig(
            name="tac2017_bigbio_kb",
            version=BIGBIO_VERSION,
            description="TAC 2017 BigBio schema",
            schema="bigbio_kb",
            subset_id="tac2017",
        ),
    ]

    DEFAULT_CONFIG_NAME = "tac2017_source"

    def _info(self) -> datasets.DatasetInfo:

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "drug": datasets.Value("string"),
                    "text": {
                        "sections": [
                            {
                                "id": datasets.Value("string"),
                                "name": datasets.Value("string"),
                                "section_text": datasets.Value("string"),
                            }
                        ],
                    },
                    "mentions": [
                        {
                            "id": datasets.Value("string"),
                            "source_id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "section_id": datasets.Value("string"),
                            "start": datasets.Value("int32"),
                            "len": datasets.Value("int32"),
                            "str": datasets.Value("string"),
                        }
                    ],
                    "relations": [
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "arg1": datasets.Value("string"),
                            "arg2": datasets.Value("string"),
                        }
                    ],
                    "reactions": [
                        {
                            "id": datasets.Value("string"),
                            "str": datasets.Value("string"),
                            "normalization": {
                                "id": datasets.Value("string"),
                                "meddra_pt": datasets.Value("string"),
                                "meddra_pt_id": datasets.Value("string"),
                                "meddra_llt": datasets.Value("string"),
                                "meddra_llt_id": datasets.Value("string"),
                            },
                        }
                    ],
                }
            )

        elif self.config.schema == "bigbio_kb":
            features = schemas.kb_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION, features=features, homepage=_HOMEPAGE, license=_LICENSE, citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        train_fpaths = {
            "train_xml/ADCETRIS.xml",
            "train_xml/BEPREVE.xml",
            "train_xml/CLEVIPREX.xml",
            "train_xml/DYSPORT.xml",
            "train_xml/FERRIPROX.xml",
            "train_xml/ILARIS.xml",
            "train_xml/KALYDECO.xml",
            "train_xml/ONFI.xml",
            "train_xml/SAPHRIS.xml",
            "train_xml/TIVICAY.xml",
            "train_xml/VIZAMYL.xml",
            "train_xml/ZYKADIA.xml",
            "train_xml/ADREVIEW.xml",
            "train_xml/BESIVANCE.xml",
            "train_xml/COARTEM.xml",
            "train_xml/EDARBI.xml",
            "train_xml/FIRAZYR.xml",
            "train_xml/IMBRUVICA.xml",
            "train_xml/KYPROLIS.xml",
            "train_xml/OTEZLA.xml",
            "train_xml/SIMPONI.xml",
            "train_xml/TOVIAZ.xml",
            "train_xml/VORAXAZE.xml",
            "train_xml/ZYTIGA.xml",
            "train_xml/AFINITOR.xml",
            "train_xml/BLINCYTO.xml",
            "train_xml/COMETRIQ.xml",
            "train_xml/ELIQUIS.xml",
            "train_xml/FULYZAQ.xml",
            "train_xml/INLYTA.xml",
            "train_xml/LUMIZYME.xml",
            "train_xml/PICATO.xml",
            "train_xml/SIRTURO.xml",
            "train_xml/TREANDA.xml",
            "train_xml/XALKORI.xml",
            "train_xml/AMPYRA.xml",
            "train_xml/BOSULIF.xml",
            "train_xml/DALVANCE.xml",
            "train_xml/ENTEREG.xml",
            "train_xml/GADAVIST.xml",
            "train_xml/INTELENCE.xml",
            "train_xml/MULTAQ.xml",
            "train_xml/POTIGA.xml",
            "train_xml/STENDRA.xml",
            "train_xml/TRULICITY.xml",
            "train_xml/XEOMIN.xml",
            "train_xml/AMYVID.xml",
            "train_xml/BREO.xml",
            "train_xml/DATSCAN.xml",
            "train_xml/EOVIST.xml",
            "train_xml/GILENYA.xml",
            "train_xml/INVOKANA.xml",
            "train_xml/NATAZIA.xml",
            "train_xml/PRADAXA.xml",
            "train_xml/STRIBILD.xml",
            "train_xml/TUDORZA.xml",
            "train_xml/XIAFLEX.xml",
            "train_xml/APTIOM.xml",
            "train_xml/CARBAGLU.xml",
            "train_xml/DIFICID.xml",
            "train_xml/ERWINAZE.xml",
            "train_xml/GILOTRIF.xml",
            "train_xml/JARDIANCE.xml",
            "train_xml/NESINA.xml",
            "train_xml/PRISTIQ.xml",
            "train_xml/TAFINLAR.xml",
            "train_xml/ULESFIA.xml",
            "train_xml/XTANDI.xml",
            "train_xml/ARCAPTA.xml",
            "train_xml/CERDELGA.xml",
            "train_xml/DOTAREM.xml",
            "train_xml/EYLEA.xml",
            "train_xml/GRANIX.xml",
            "train_xml/JEVTANA.xml",
            "train_xml/NEURACEQ.xml",
            "train_xml/PROLIA.xml",
            "train_xml/TANZEUM.xml",
            "train_xml/ULORIC.xml",
            "train_xml/YERVOY.xml",
            "train_xml/BELEODAQ.xml",
            "train_xml/CHOLINE.xml",
            "train_xml/DUAVEE.xml",
            "train_xml/FANAPT.xml",
            "train_xml/HALAVEN.xml",
            "train_xml/JUBLIA.xml",
            "train_xml/NORTHERA.xml",
            "train_xml/PROMACTA.xml",
            "train_xml/TECFIDERA.xml",
            "train_xml/VICTRELIS.xml",
            "train_xml/ZERBAXA.xml",
            "train_xml/BENLYSTA.xml",
            "train_xml/CIMZIA.xml",
            "train_xml/DUREZOL.xml",
            "train_xml/FARXIGA.xml",
            "train_xml/HORIZANT.xml",
            "train_xml/KALBITOR.xml",
            "train_xml/NULOJIX.xml",
            "train_xml/QUTENZA.xml",
            "train_xml/TEFLARO.xml",
            "train_xml/VIMIZIM.xml",
            "train_xml/ZYDELIG.xml",
        }

        urls = _URLS[_DATASETNAME]
        data_dir = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"files": [os.path.join(data_dir, path) for path in train_fpaths], "split": "train"},
            ),
        ]

    def _generate_example_sections(self, uid, source_sections_tree):
        """
        Parse sections XML

        Parameters
        ----------
        uid : int
            unique identifier being updated with each execution
        source_sections_tree : etree object
            XML of drug label sections

        Returns
        ----------
        int
            updated unique identifier
        dict
            drug label sections information
        """

        sections = []

        for source_section in source_sections_tree:
            source_section_id = source_section.attrib["id"]
            source_section_name = source_section.attrib["name"]
            source_section_text = source_section.text
            section = {"id": source_section_id, "name": source_section_name, "section_text": source_section_text}
            sections.append(section)
            uid += 1

        return (uid, sections)

    def _generate_example_mentions(self, uid, source_mentions_tree):
        """
        Parse mentions XML

        Parameters
        ----------
        uid : int
            unique identifier being updated with each execution
        source_mentions_tree : etree object
            XML of drug label ADR mentions

        Returns
        ----------
        int
            updated unique identifier
        dict
            ADR mentions information
        """
        mentions = []
        for source_mention in source_mentions_tree:
            source_mention_id = source_mention.attrib["id"]
            source_mention_section_id = source_mention.attrib["section"]
            source_mention_type = source_mention.attrib["type"]
            source_mention_start = source_mention.attrib["start"]
            source_mention_len = source_mention.attrib["len"]
            source_mention_str = source_mention.attrib["str"]

            if "," in source_mention_start:
                source_mention_starts = source_mention_start.split(",")
                source_mention_lens = source_mention_len.split(",")
                i = 0
                for source_mention_start in source_mention_starts:
                    mention = {
                        "id": str(uid),
                        "source_id": source_mention_id,
                        "type": source_mention_type,
                        "section_id": source_mention_section_id,
                        "start": int(source_mention_start),
                        "len": int(source_mention_lens[i]),
                        "str": source_mention_str,
                    }
                    mentions.append(mention)
                    uid += 1
                    i+=1
            else:
                mention = {
                    "id": str(uid),
                    "source_id": source_mention_id,
                    "type": source_mention_type,
                    "section_id": source_mention_section_id,
                    "start": int(source_mention_start),
                    "len": int(source_mention_len),
                    "str": source_mention_str,
                }
                mentions.append(mention)
                uid += 1
        return (uid, mentions)

    def _generate_example_relations(self, uid, source_relations_tree):
        """
        Parse relations XML

        Parameters
        ----------
        uid : int
            unique identifier being updated with each execution
        source_relations_tree : etree object
            XML of drug label ADR relations

        Returns
        ----------
        int
            updated unique identifier
        dict
            drug label relations information
        """
        relations = []
        for source_relation in source_relations_tree:
            source_relation_id = source_relation.attrib["id"]
            source_relation_type = source_relation.attrib["type"]
            source_relation_arg1 = source_relation.attrib["arg1"]
            source_relation_arg2 = source_relation.attrib["arg2"]

            relation = {
                "id": source_relation_id,
                "type": source_relation_type,
                "arg1": source_relation_arg1,
                "arg2": source_relation_arg2,
            }
            relations.append(relation)
            uid += 1
        return (uid, relations)

    def _generate_example_reactions(self, uid, source_reactions_tree):
        """
        Parse reactions XML

        Parameters
        ----------
        uid : int
            unique identifier being updated with each execution
        source_reactions_tree : etree object
            XML of drug label reaction normalizations to MedDRA terms

        Returns
        ----------
        int
            updated unique identifier
        dict
            reactions normalization information
        """
        reactions = []
        for source_reaction in source_reactions_tree:
            source_reaction_id = source_reaction.attrib["id"]
            source_reaction_str = source_reaction.attrib["str"]

            reaction = {"id": source_reaction_id, "str": source_reaction_str}

            for source_reaction_normalization in source_reaction:
                source_reaction_normalization_id = source_reaction_normalization.attrib["id"]

                normalization = {"id": source_reaction_normalization_id}

                if "meddra_pt" in source_reaction_normalization.attrib:
                    source_reaction_normalization_meddra_pt = source_reaction_normalization.attrib["meddra_pt"]
                    source_reaction_normalization_meddra_pt_id = source_reaction_normalization.attrib["meddra_pt_id"]
                    normalization["meddra_pt"] = source_reaction_normalization_meddra_pt
                    normalization["meddra_pt_id"] = source_reaction_normalization_meddra_pt_id
                else:
                    normalization["meddra_pt"] = ""
                    normalization["meddra_pt_id"] = ""

                if "meddra_llt" in source_reaction_normalization.attrib:
                    source_reaction_normalization_meddra_llt = source_reaction_normalization.attrib["meddra_llt"]
                    source_reaction_normalization_meddra_llt_id = source_reaction_normalization.attrib["meddra_llt_id"]
                    normalization["meddra_llt"] = source_reaction_normalization_meddra_llt
                    normalization["meddra_llt_id"] = source_reaction_normalization_meddra_llt_id
                else:
                    normalization["meddra_llt"] = ""
                    normalization["meddra_llt_id"] = ""
                reaction["normalization"] = normalization
                uid += 1
            reactions.append(reaction)
            uid += 1
        return (uid, reactions)

    def _generate_example_kb_passages(self, uid, drug_name, source_sections_tree):
        """
        Parse sections XML into passages for KB schema

        Parameters
        ----------
        uid : int
            unique identifier being updated with each execution
        source_sections_tree : etree object
            XML of drug label sections

        Returns
        ----------
        int
            updated unique identifier
        dict
            KB schema passages information
        """
        passages = []

        overall_text = ""

        for source_section in source_sections_tree:
            passage_id = drug_name + "_" + source_section.attrib["id"]
            passage_type = source_section.attrib["name"]
            passage_text = source_section.text
            
            passage_offsets = (len(overall_text), len(passage_text)+len(overall_text))
            passage = {"id": passage_id, "type": passage_type, "text": [passage_text], "offsets": [passage_offsets]}
            passages.append(passage)
            uid += 1
            overall_text = overall_text+passage_text
            
        return (uid, passages)

    def _generate_example_kb_entities(self, uid, drug_name, source_mentions_tree, normalizations, passages):
        """
        Parse mentions XML into entities for KB schema, including normalizations from source "reactions" data

        Parameters
        ----------
        uid : int
            unique identifier being updated with each execution
        source_mentions_tree : etree object
            XML of drug label ADR mentions

        Returns
        ----------
        int
            updated unique identifier
        dict
            KB schema entities information
        """
        entities = []
        for source_mention in source_mentions_tree:
            entity_type = source_mention.attrib["type"]
            entity_text = source_mention.attrib["str"]
            entity_id = source_mention.attrib["id"]
            passage = source_mention.attrib["section"]

            passage_num = int(passage[1:])

            pretext = ""

            if passage_num>1:
                texts = [passage["text"][0] for passage in passages for i in range(0,passage_num-1) if passage["id"] == "S"+str(i)]
                pretext = "".join(text for text in texts)
                
            relevant_normalizations = [
                normalization for normalization in normalizations if entity_text == normalization["str"]
            ]

            source_mention_start = source_mention.attrib["start"]
            source_mention_len = source_mention.attrib["len"]

            if "," in source_mention_start:
                source_mention_starts = source_mention_start.split(",")
                source_mention_lens = source_mention_len.split(",")

                for source_mention_start in source_mention_starts:
                    entity = {
                        "id": drug_name + "_" + entity_id,
                        "type": entity_type,
                        "text": [entity_text],
                        "offsets": [
                            [len(pretext) + int(source_mention_start), len(pretext) + int(source_mention_start) + int(source_mention_lens[0])]
                        ],
                        "normalized": [
                            {"db_name": "meddra", "db_id": rn["meddra_id"]} for rn in relevant_normalizations
                        ],
                    }
                    entities.append(entity)
                    uid += 1
            else:
                entity = {
                    "id": drug_name + "_" + entity_id,
                    "type": entity_type,
                    "text": [entity_text],
                    "offsets": [[len(pretext) + int(source_mention_start), len(pretext) + int(source_mention_start) + int(source_mention_len)]],
                    "normalized": [{"db_name": "meddra", "db_id": rn["meddra_id"]} for rn in relevant_normalizations],
                }
                entities.append(entity)
                uid += 1
        return (uid, entities)

    def _generate_kb_entity_normalizations(self, uid, source_reactions_tree):
        """
        Parse reactions XML into entity normalizations for KB schema

        Parameters
        ----------
        uid : int
            unique identifier being updated with each execution
        source_reactions_tree : etree object
            XML of drug label reaction normalizations to MedDRA terms

        Returns
        ----------
        int
            updated unique identifier
        dict
            reactions normalization information for KB schema
        """
        normalizations = []
        for source_reaction in source_reactions_tree:

            for source_reaction_normalization in source_reaction:
                if "meddra_pt" in source_reaction_normalization.attrib:
                    normalized_term = {}
                    normalized_term["str"] = source_reaction.attrib["str"]
                    source_reaction_normalization_meddra_pt_id = source_reaction_normalization.attrib["meddra_pt_id"]
                    normalized_term["meddra_id"] = source_reaction_normalization_meddra_pt_id
                    normalizations.append(normalized_term)

                if "meddra_llt" in source_reaction_normalization.attrib:
                    normalized_term = {}
                    normalized_term["str"] = source_reaction.attrib["str"]
                    source_reaction_normalization_meddra_llt_id = source_reaction_normalization.attrib["meddra_llt_id"]
                    normalized_term["meddra_id"] = source_reaction_normalization_meddra_llt_id
                    normalizations.append(normalized_term)
        return(uid, normalizations)

    def _generate_example_kb_relations(self, uid, drug_name, source_relations_tree):
        """
        Parse relations XML for KB schema

        Parameters
        ----------
        uid : int
            unique identifier being updated with each execution
        source_relations_tree : etree object
            XML of drug label ADR relations

        Returns
        ----------
        int
            updated unique identifier
        dict
            drug label relations information for KB schema
        """
        relations = []
        for source_relation in source_relations_tree:
            source_relation_id = source_relation.attrib["id"]
            source_relation_type = source_relation.attrib["type"]
            source_relation_arg1 = source_relation.attrib["arg1"]
            source_relation_arg2 = source_relation.attrib["arg2"]

            relation = {
                "id": source_relation_id,
                "type": source_relation_type,
                "arg1_id": drug_name + "_" + source_relation_arg1,
                "arg2_id": drug_name + "_" + source_relation_arg2,
                "normalized": [],
            }
            uid += 1
            relations.append(relation)
        return (uid, relations)

    def _generate_examples(self, files, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""
        uid = 0

        for file in files:
            with open(file) as xml_file:
                source_tree = ET.parse(xml_file)
                source_label = source_tree.getroot()
                source_drug_name = source_label.attrib["drug"]

                source_text = [source_child for source_child in source_label if source_child.tag == "Text"][0]
                source_mentions = [source_child for source_child in source_label if source_child.tag == "Mentions"][0]
                source_relations = [source_child for source_child in source_label if source_child.tag == "Relations"][
                    0
                ]
                source_reactions = [source_child for source_child in source_label if source_child.tag == "Reactions"][
                    0
                ]

                if self.config.schema == "source":
                    example = {"drug": source_drug_name, "text": {}, "mentions": [], "relations": [], "reactions": []}
                    uid, sections = self._generate_example_sections(uid, source_text)
                    example["text"] = {"sections": sections}

                    uid, mentions = self._generate_example_mentions(uid, source_mentions)
                    example["mentions"] = mentions

                    uid, relations = self._generate_example_relations(uid, source_relations)
                    example["relations"] = relations

                    uid, reactions = self._generate_example_reactions(uid, source_reactions)
                    example["reactions"] = reactions
                    yield uid, example

                elif self.config.schema == "bigbio_kb":

                    example = {
                        "id": uid,
                        "document_id": source_drug_name,
                        "passages": [],
                        "entities": [],
                        "relations": [],
                        "events": [],
                        "coreferences": [],
                    }

                    uid, entity_normalizations = self._generate_kb_entity_normalizations(uid, source_reactions)

                    uid, passages = self._generate_example_kb_passages(uid, source_drug_name, source_text)
                    example["passages"] = passages

                    uid, entities = self._generate_example_kb_entities(
                        uid, source_drug_name, source_mentions, entity_normalizations, passages
                    )
                    example["entities"] = entities

                    uid, relations = self._generate_example_kb_relations(uid, source_drug_name, source_relations)
                    example["relations"] = relations

                    yield uid, example
