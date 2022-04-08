# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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


from pathlib import Path
from typing import List

import datasets
from utils import parsing, schemas
from utils.configs import BigBioConfig
from utils.constants import Tasks


_DATASETNAME = "bionlp_st_2013_gro"
_SOURCE_VIEW_NAME = "source"
_UNIFIED_VIEW_NAME = "bigbio"

_CITATION = """\
@inproceedings{kim-etal-2013-gro,
    title = "{GRO} Task: Populating the Gene Regulation Ontology with events and relations",
    author = "Kim, Jung-jae  and
      Han, Xu  and
      Lee, Vivian  and
      Rebholz-Schuhmann, Dietrich",
    booktitle = "Proceedings of the {B}io{NLP} Shared Task 2013 Workshop",
    month = aug,
    year = "2013",
    address = "Sofia, Bulgaria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W13-2007",
    pages = "50--57",
}
"""

_DESCRIPTION = """\
GRO Task: Populating the Gene Regulation Ontology with events and
relations. A data set from the bio NLP shared tasks competition from 2013
"""

_HOMEPAGE = "https://github.com/openbiocorpora/bionlp-st-2013-gro"

_LICENSE = "GENIA Project License for Annotated Corpora:" \
           "https://github.com/openbiocorpora/bionlp-st-2011-epi/blob/master/LICENSE"

_URLs = {"source": "https://github.com/openbiocorpora/bionlp-st-2013-gro/archive/refs/heads/master.zip",
         "bigbio_kb": "https://github.com/openbiocorpora/bionlp-st-2013-gro/archive/refs/heads/master.zip",}

_SUPPORTED_TASKS = [Tasks.EVENT_EXTRACTION,
                    Tasks.NAMED_ENTITY_RECOGNITION,
                    Tasks.RELATION_EXTRACTION,
                    ]
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"


class bionlp_st_2013_gro(datasets.GeneratorBasedBuilder):
    """GRO Task: Populating the Gene Regulation Ontology with events and
    relations. A data set from the bio NLP shared tasks competition from 2013"""

    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="bionlp_st_2013_gro_source",
            version=SOURCE_VERSION,
            description="bionlp_st_2013_gro source schema",
            schema="source",
            subset_id="bionlp_st_2013_gro",
        ),
        BigBioConfig(
            name="bionlp_st_2013_gro_bigbio_kb",
            version=BIGBIO_VERSION,
            description="bionlp_st_2013_gro BigBio schema",
            schema="bigbio_kb",
            subset_id="bionlp_st_2013_gro",
        ),
    ]

    DEFAULT_CONFIG_NAME = "bionlp_st_2013_gro_source"

    _ENTITY_TYPES = {
         'Eukaryote',
         'Function',
         'CellComponent',
         'Nucleus',
         'Phenotype',
         'ZincCoordinatingDomainTF',
         'MutantProtein',
         'Gene',
         'TranscriptionActivator',
         'TranscriptionRepressor',
         'Protein',
         'Cell',
         'Enzyme',
         'Stress',
         'Tissue',
         'MutantGene',
         'LivingEntity',
         'ExperimentalMethod',
         'HMGTF',
         'Promoter',
         'DNABindingDomainOfProtein',
         'TranscriptionCoactivator',
         'Peptide',
         'bHLHTF',
         'DNARegion',
         'AminoAcid',
         'ProteinDomain',
         'TranscriptionRegulator',
         'Heterochromatin',
         'Chromosome',
         'Chromatin',
         'Plant',
         'DNA',
         'BindingAssay',
         'RibosomalDNA',
         'OpenReadingFrame',
         'EukaryoticCell',
         'DNAFragment',
         'Sequence',
         'TranscriptionFactor',
         'HMG',
         'ProteinSubunit',
         'TranscriptionFactorComplex',
         'ProteinComplex',
         'TranscriptionRepressorActivity',
         'ZincCoordinatingDomain',
         'Ion',
         'TranscriptionCorepressor',
         'MolecularStructure',
         'Gas',
         'ReporterGene',
         'TranslationFactor',
         'Lipid',
         'FusionProtein',
         'Transcript',
         'ProteinBindingSiteOfDNA',
         'ProteinIdentification',
         'Locus',
         'ProteinCodingRegion',
         'OrganicChemical',
         'ComplexOfProteinAndRNA',
         'Spliceosome',
         'CatalyticActivity',
         'Nucleotide',
         'StructuralDomainOfTranscriptionRepressor',
         'Silencer',
         'Histone',
         'Ligand',
         'Agonist',
         'GeneralTranscriptionFactor',
         'TranscriptionCofactor',
         'TranscriptionFactorActivity',
         'MutantRegulatoryDNARegion',
         'FusionOfGeneWithReporterGene',
         'RegulatoryDNARegion',
         'PositiveTranscriptionRegulator',
         'NegativeTranscriptionRegulator',
         'UpstreamRegulatorySequence',
         'Enhancer',
         'GeneRegion',
         'TBP',
         'HistoneFold',
         'HomeoboxTF',
         'TranscriptionFactorBindingSiteOfDNA',
         'Exon',
         'DNABindingDomainOfTF',
         'Chemical',
         'MessengerRNA',
         'SmallInterferingRNA',
         'RNAPolymeraseIIHoloenzyme',
         'bHLH',
         'bZIP',
         'TemporalQuality',
         'NonPhysicalContinuant',
         'ExpressionProfiling',
         'RNA',
         'DNAMolecule',
         'MolecularEntity',
         'ReporterGeneConstruction',
         'Substrate',
         'PhysicalContinuant',
         'Orientation',
         'CpGIsland',
         'NuclearReceptor',
         'TranscriptionInitiationComplex',
         'RNAPolymeraseII',
         'PhosphateGroup',
         'RNAPolymerase_II_Promoter',
         'Antagonist',
         'ChromosomalDNA',
         'Virus',
         'Carbohydrate',
         'Quality',
         'RNAPolymerase',
         'ComplexOfProteinAndDNA',
         'ThreeDimensionalMolecularStructure',
         'MicroRNA',
         'Posttranscriptional',
         'GeneExpressionRegulator',
         'FeedbackLoop',
         'Epigenetic',
         'ProteinBindingSiteOfProtein',
         'OxidativeStress',
         'DoubleStrandDNA',
         'SingleStrandDNA',
         'Posttranslational',
         'TranscriptionActivatorActivity',
         'Intron',
         'BasicDomain',
         'Solution',
         'Nucleosome',
         'BindingSiteOfProtein',
         'NcRNA',
         'X_Chromosome',
         'ComplexMolecularEntity',
         'RNARegion',
         'SequenceHomologyAnalysis',
         'bZIPTF',
         'GeneProduct',
         'ProteinCodingDNARegion',
         'Bacterium',
         'TATAbox',
         'Temperature',
         'HistoneFoldTF',
         'PrimaryStructure',
         'Microorganism',
         'Nucleoside',
         'Holoenzyme',
         'NucleicAcid',
         'InorganicChemical',
         'Conformation',
         'Prokaryote',
         'AntisenseRNA',
         'tRNA',
         'SecondMessenger',
         'MutatedProtein',
         'MolecularFunction',
         'Ribosome',
         'ForkheadWingedHelixTF',
         'Attenuator',
         'DNABindingSite',
         'AP2EREBPRelatedDomain',
         'ForkheadWingedHelix',
         'HomeoBox',
         'FusionGene',
         'RibosomalRNA',
         'RuntLikeDomain',
         'PhysicalContact',
         'Operon',
         'Polymerase',
         'BetaScaffoldDomain_WithMinorGrooveContacts',
         'RNAMolecule',
         'Vitamin',
         'Organism',
    }

    def _info(self):
        """
        - `features` defines the schema of the parsed data set. The schema depends on the
        chosen `config`: If it is `_SOURCE_VIEW_NAME` the schema is the schema of the
        original data. If `config` is `_UNIFIED_VIEW_NAME`, then the schema is the
        canonical KB-task schema defined in `biomedical/schemas/kb.py`.
        """
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
                    "events": [  # E line in brat
                        {
                            "trigger": datasets.Value(
                                "string"
                            ),  # refers to the text_bound_annotation of the trigger,
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "arguments": datasets.Sequence(
                                {
                                    "role": datasets.Value("string"),
                                    "ref_id": datasets.Value("string"),
                                }
                            ),
                        }
                    ],
                    "relations": [  # R line in brat
                        {
                            "id": datasets.Value("string"),
                            "head": {
                                "ref_id": datasets.Value("string"),
                                "role": datasets.Value("string"),
                            },
                            "tail": {
                                "ref_id": datasets.Value("string"),
                                "role": datasets.Value("string"),
                            },
                            "type": datasets.Value("string"),
                        }
                    ],
                    "equivalences": [  # Equiv line in brat
                        {
                            "id": datasets.Value("string"),
                            "ref_ids": datasets.Sequence(datasets.Value("string")),
                        }
                    ],
                    "attributes": [  # M or A lines in brat
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "ref_id": datasets.Value("string"),
                            "value": datasets.Value("string"),
                        }
                    ],
                    "normalizations": [  # N lines in brat
                        {
                            "id": datasets.Value("string"),
                            "type": datasets.Value("string"),
                            "ref_id": datasets.Value("string"),
                            "resource_name": datasets.Value(
                                "string"
                            ),  # Name of the resource, e.g. "Wikipedia"
                            "cuid": datasets.Value(
                                "string"
                            ),  # ID in the resource, e.g. 534366
                            "text": datasets.Value(
                                "string"
                            ),  # Human readable description/name of the entity, e.g. "Barack Obama"
                        }
                    ],
                },
            )
        elif self.config.schema == "bigbio_kb":
            features = schemas.kb_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:

        my_urls = _URLs[self.config.schema]
        data_dir = Path(dl_manager.download_and_extract(my_urls))
        data_files = {
            "train": data_dir / f'bionlp-st-2013-gro-master' / "original-data" / "train",
            "dev": data_dir / f'bionlp-st-2013-gro-master' / "original-data" / "devel",
            "test": data_dir / f'bionlp-st-2013-gro-master' / "original-data" / "test",
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_files": data_files["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"data_files": data_files["dev"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"data_files": data_files["test"]},
            ),
        ]

    def _generate_examples(self, data_files: Path):
        if self.config.schema == "source":
            txt_files = list(data_files.glob("*txt"))
            for guid, txt_file in enumerate(txt_files):
                example = parsing.parse_brat_file(txt_file)
                example["id"] = str(guid)
                yield guid, example
        elif self.config.schema == "bigbio_kb":
            txt_files = list(data_files.glob("*txt"))
            for guid, txt_file in enumerate(txt_files):
                example = parsing.brat_parse_to_bigbio_kb(
                    parsing.parse_brat_file(txt_file),
                    entity_types=self._ENTITY_TYPES
                )
                example["id"] = str(guid)
                yield guid, example
        else:
            raise ValueError(f"Invalid config: {self.config.name}")

