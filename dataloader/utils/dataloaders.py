"""
TODO: Ask Leon about cache_path in Linneaus
"""
import os
from pathlib import Path
from typing import List, Optional, Dict

import flair.tokenization
import pybrat.parser
from flair.datasets import biomedical as hunflair
from pybrat.parser import BratParser

from dataloader.utils.downloaders import download_data, uncompress_data, chemprot_2_standoff


def _write_hunflair_internal_to_brat(
    dataset: hunflair.InternalBioNerDataset, brat_dir: Path
):
    brat_dir.mkdir(parents=True, exist_ok=True)
    for document, text in dataset.documents.items():
        with (brat_dir / document).with_suffix(".ann").open("w") as f:
            for i, entity in enumerate(
                dataset.entities_per_document[document], start=1
            ):
                mention = text[entity.char_span.start : entity.char_span.stop]
                f.write(
                    f"T{i}\t{entity.type} {entity.char_span.start} {entity.char_span.stop}\t{mention}\n"
                )
        with (brat_dir / document).with_suffix(".txt").open("w") as f:
            f.write(text.replace(hunflair.SENTENCE_TAG, " "))


class BC5CDR:
    """ """

    def __init__(self, data_root: str, parser):
        """ """
        self.train_file = (
            Path(data_root)
            / "CDR_Data"
            / "CDR.Corpus.v010516"
            / "CDR_TrainingSet.BioC.xml"
        )
        self.dev_file = (
            Path(data_root)
            / "CDR_Data"
            / "CDR.Corpus.v010516"
            / "CDR_DevelopmentSet.BioC.xml"
        )
        self.test_file = (
            Path(data_root) / "CDR_Data" / "CDR.Corpus.v010516" / "CDR_TestSet.BioC.xml"
        )

        # Download data if not available
        if not list((Path(data_root) / "CDR_Data" / "CDR.Corpus.v010516").glob("*")):
            self._download(
                Path(data_root),
                self.train_file,
                self.dev_file,
                self.test_file,
            )

        self.train = parser.parse(str(self.train_file))
        self.test = parser.parse(str(self.dev_file))
        self.dev = parser.parse(str(self.test_file))

    @staticmethod
    def _download(data_root, train_file, dev_file, test_file):
        """Download the BC5CDR Dataset"""
        _URL = "http://www.biocreative.org/media/store/files/2016/CDR_Data.zip"

        # Make data root, if not present
        if not os.path.exists(data_root):
            os.makedirs(data_root)

        outpath = data_root / _URL.split("/")[-1]
        download_data(_URL, outpath)
        uncompress_data(outpath, data_root.resolve())


class ChemProt:
    """
    ChemProt Corpus (deprecated, the newest version is now DrugProt)

    """

    def __init__(self, data_root: str, parser):

        # Directory paths
        self.train_dir = Path(data_root) / "ChemProt_Corpus" / "chemprot_training"
        self.test_dir = Path(data_root) / "ChemProt_Corpus" / "chemprot_test_gs"
        self.dev_dir = Path(data_root) / "ChemProt_Corpus" / "chemprot_development"
        self.sample_dir = Path(data_root) / "ChemProt_Corpus" / "chemprot_sample"

        # Download data if not available
        if (
            not list(self.train_dir.glob("*"))
            or not list(self.test_dir.glob("*"))
            or not list(self.dev_dir.glob("*"))
            or not list(self.sample_dir.glob("*"))
        ):
            self._download(
                Path(data_root),
                self.train_dir,
                self.test_dir,
                self.dev_dir,
                self.sample_dir,
            )

        self.train = parser.parse(self.train_dir / "brat")
        self.test = parser.parse(self.test_dir / "brat")
        self.dev = parser.parse(self.dev_dir / "brat")
        self.sample = parser.parse(self.sample_dir / "brat")

    @staticmethod
    def _download(data_root, train_dir, test_dir, dev_dir, sample_dir):
        _URL = "https://biocreative.bioinformatics.udel.edu/media/store/files/2017/ChemProt_Corpus.zip"

        # Make data root, if not present
        if not os.path.exists(data_root):
            os.makedirs(data_root)

        outpath = data_root / _URL.split("/")[-1]

        download_data(_URL, outpath)
        uncompress_data(outpath, data_root.resolve())

        # Chemprot needs to be converted to brat format
        # Each split also needs to be unzipped
        for split in [train_dir, test_dir, dev_dir, sample_dir]:

            # Unzip datasplit
            uncompress_data(str(split) + ".zip", data_root / "ChemProt_Corpus")

            # Convert to brat
            brat_path = str(split / "brat")
            chemprot_2_standoff(str(split), brat_path)


class DDI:
    """
    DDI Corpus

    For further information, see:
        María Herrero-Zazo, Isabel Segura-Bedmar, Paloma Martínez, Thierry Declerck, The DDI corpus:
        An annotated corpus with pharmacological substances and drug–drug interactions, Journal of Biomedical Informatics,
        Volume 46, Issue 5, October 2013, Pages 914-920, http://dx.doi.org/10.1016/j.jbi.2013.07.011.

    """

    def __init__(self, data_root: str, parser):

        # Directory paths
        self.train_dir = Path(data_root) / "DDICorpusBrat" / "Train"
        self.test_dir = Path(data_root) / "DDICorpusBrat" / "Test"

        # Download data if not available
        if not list(self.train_dir.glob("*")) or not list(self.test_dir.glob("*")):
            self._download(Path(data_root), self.train_dir, self.test_dir)

        self.train = parser.parse(self.train_dir)
        self.test = parser.parse(self.test_dir)

    @staticmethod
    def _download(data_root, train_dir, test_dir):
        _URL = (
            "https://github.com/isegura/DDICorpus/raw/master/DDICorpus-2013(BRAT).zip"
        )

        # Make data root, if not present
        if not os.path.exists(data_root):
            os.makedirs(data_root)

        outpath = data_root / _URL.split("/")[-1]

        download_data(_URL, outpath)
        uncompress_data(outpath, data_root.resolve())


class JNLPBA:
    """
    Original corpus of the JNLPBA shared task.
    For further information see Kim et al.:
      Introduction to the Bio-Entity Recognition Task at JNLPBA
      https://www.aclweb.org/anthology/W04-1213.pdf
    """

    def __init__(self, data_root: str, parser):

        # Directory paths
        self.train_dir = Path(data_root) / "JNLPBA" / "train"
        self.test_dir = Path(data_root) / "JNLPBA" / "test"
        cache_path = Path(data_root) / "tmp"

        # Download data if not available
        if not list(self.train_dir.glob("*")) or not list(self.test_dir.glob("*")):
            self._download(cache_path, self.train_dir, self.test_dir)

        self.train = parser.parse(self.train_dir)
        self.test = parser.parse(self.test_dir)

    @staticmethod
    def _download(cache_path, train_dir, test_dir):
        train_dataset = hunflair.HunerJNLPBA.download_and_prepare_train(
            data_folder=cache_path / "JNLPBA", sentence_tag=" "
        )
        test_dataset = hunflair.HunerJNLPBA.download_and_prepare_test(
            data_folder=cache_path / "JNLPBA", sentence_tag=" "
        )
        _write_hunflair_internal_to_brat(train_dataset, train_dir)
        _write_hunflair_internal_to_brat(test_dataset, test_dir)


class CellFinder:
    """
    Original CellFinder corpus containing cell line, species and gene annotations.
    For futher information see Neves et al.:
        Annotating and evaluating text for stem cell research
        https://pdfs.semanticscholar.org/38e3/75aeeeb1937d03c3c80128a70d8e7a74441f.pdf
    """

    def __init__(self, data_root, parser):
        # Directory paths
        self.train_dir = Path(data_root) / "cellfinder" / "train"
        cache_path = Path(data_root) / "tmp"

        # Download data if not available
        if not list(self.train_dir.glob("*")):
            self._download(cache_path, self.train_dir)
        self.train = parser.parse(self.train_dir)

    @staticmethod
    def _download(cache_path, train_dir):
        train_dataset = hunflair.CELL_FINDER.download_and_prepare(
            data_folder=cache_path / "cellfinder"
        )
        _write_hunflair_internal_to_brat(train_dataset, train_dir)


class Linneaus:
    """
    Original LINNEAUS corpus containing species annotations.
    For further information see Gerner et al.:
         LINNAEUS: a species name identification system for biomedical literature
         https://www.ncbi.nlm.nih.gov/pubmed/20149233
    """

    def __init__(self, data_root, parser):
        # Directory paths
        self.train_dir = Path(data_root) / "linneaus" / "train"
        cache_path = Path(data_root) / "tmp"

        # Download data if not available
        if not list(self.train_dir.glob("*")):
            self._download(cache_path, self.train_dir)
        self.train = parser.parse(self.train_dir)

    @staticmethod
    def _download(cache_path, train_dir):
        # Linneaus appears sensitive to the cache_path being defined
        if not os.path.exists(cache_path):
            os.makedirs(str(cache_path))

        train_dataset = hunflair.LINNEAUS.download_and_parse_dataset(
            data_dir=cache_path / "linneaus"
        )
        _write_hunflair_internal_to_brat(train_dataset, train_dir)


class HunFlairDataset:

    name_to_class = {
        # CELLLINE
        "hunflair_cellline_cll": hunflair.HUNER_CELL_LINE_CLL,
        "hunflair_cellline_cellfinder": hunflair.HUNER_CELL_LINE_CELL_FINDER,
        "hunflair_cellline_gellus": hunflair.HUNER_CELL_LINE_GELLUS,
        "hunflair_cellline_jnlpba": hunflair.HUNER_CELL_LINE_JNLPBA,
        # CHEMICAL
        "hunflair_chemical_chebi": hunflair.HUNER_CHEMICAL_CHEBI,
        "hunflair_chemical_cdr": hunflair.HUNER_CHEMICAL_CDR,
        "hunflair_chemical_cemp": hunflair.HUNER_CHEMICAL_CEMP,
        "hunflair_chemical_scai": hunflair.HUNER_CHEMICAL_SCAI,
        # OFFLINE "hunflair_chemical_bionlp2013_cg": hunflair.HUNER_CHEMICAL_BIONLP2013_CG,
        # DISEASE
        "hunflair_disease_cdr": hunflair.HUNER_DISEASE_CDR,
        "hunflair_disease_pdr": hunflair.HUNER_DISEASE_PDR,
        "hunflair_disease_ncbi": hunflair.HUNER_DISEASE_NCBI,
        "hunflair_disease_scai": hunflair.HUNER_DISEASE_SCAI,
        "hunflair_disease_mirna": hunflair.HUNER_DISEASE_MIRNA,
        #OFFLINE "hunflair_disease_bionlp2013_cg" : hunflair.HUNER_DISEASE_BIONLP2013_CG,
        "hunflair_disease_variome": hunflair.HUNER_DISEASE_VARIOME,
        # GENE/PROTEIN
        "hunflair_gene_variome": hunflair.HUNER_GENE_VARIOME,
        "hunflair_gene_fsu": hunflair.HUNER_GENE_FSU,
        "hunflair_gene_gpro": hunflair.HUNER_GENE_GPRO,
        "hunflair_gene_deca": hunflair.HUNER_GENE_DECA,
        "hunflair_gene_iepa": hunflair.HUNER_GENE_IEPA,
        "hunflair_gene_bc2gm": hunflair.HUNER_GENE_BC2GM,
        "hunflair_gene_bio_infer": hunflair.HUNER_GENE_BIO_INFER,
        #OFFLINE "hunflair_gene_bionlp2013_cg": hunflair.HUNER_GENE_BIONLP2013_CG,
        "hunflair_gene_cellfinder": hunflair.HUNER_GENE_CELL_FINDER,
        "hunflair_gene_chebi": hunflair.HUNER_GENE_CHEBI,
        "hunflair_gene_craftv4": hunflair.HUNER_GENE_CRAFT_V4,
        "hunflair_gene_jnlpba": hunflair.HUNER_GENE_JNLPBA,
        "hunflair_gene_loctext": hunflair.HUNER_GENE_LOCTEXT,
        "hunflair_gene_mirna": hunflair.HUNER_GENE_MIRNA,
        "hunflair_gene_osiris": hunflair.HUNER_GENE_OSIRIS,
        # SPECIES
        "hunflair_species_cellfinder": hunflair.HUNER_SPECIES_CELL_FINDER,
        "hunflair_species_s800": hunflair.HUNER_SPECIES_S800,
        "hunflair_species_chebi": hunflair.HUNER_SPECIES_CHEBI,
        "hunflair_species_mirna": hunflair.HUNER_SPECIES_MIRNA,
        "hunflair_species_loctext": hunflair.HUNER_SPECIES_LOCTEXT,
        # OFFLINE "hunflair_species_bionlp2013_cg": hunflair.HUNER_SPECIES_BIONLP2013_CG,
        "hunflair_species_craftv4": hunflair.HUNER_SPECIES_CRAFT_V4,
        "hunflair_species_linneaus": hunflair.HUNER_SPECIES_LINNEAUS,
        "hunflair_species_variome": hunflair.HUNER_SPECIES_VARIOME,

    }

    def __init__(self, data_root: Path, parser, name: str):
        if name not in self.name_to_class:
            raise ValueError(f"`name' has to be one of {', '.join(self.name_to_class.keys())}")

        self.train_dir = Path(data_root) / name / "train"
        self.dev_dir = Path(data_root) / name / "dev"
        self.test_dir = Path(data_root) / name / "test"

        if not list(self.train_dir.glob("*")) or not list(self.dev_dir.glob("*")) or not list(self.test_dir.glob("*")):
            self._download(name, data_root=Path(data_root))

        self.train = parser.parse(self.train_dir)
        self.dev = parser.parse(self.dev_dir)
        self.test = parser.parse(self.test_dir)

    def _download(self, name: str, data_root: Path):
        hunflair_dataset: hunflair.HunerDataset = self.name_to_class[name](base_path=data_root/"flair")
        hunflair_dataset_name = hunflair_dataset.__class__.__name__.lower()

        internal_dataset = hunflair_dataset.to_internal(data_root / "flair" / hunflair_dataset_name)
        train_data = hunflair_dataset.get_subset(internal_dataset, "train", split_dir=data_root / "flair" / hunflair_dataset_name / "splits")
        dev_data = hunflair_dataset.get_subset(internal_dataset, "dev", split_dir=data_root / "flair" / hunflair_dataset_name / "splits")
        test_data = hunflair_dataset.get_subset(internal_dataset, "test", split_dir=data_root / "flair" / hunflair_dataset_name / "splits")

        _write_hunflair_internal_to_brat(train_data, self.train_dir)
        _write_hunflair_internal_to_brat(dev_data, self.dev_dir)
        _write_hunflair_internal_to_brat(test_data, self.test_dir)


class CustomDataset:
    """
    Enables parsing for a custom dataset that is not already included in the dataloaders.
    """

    def __init__(self, data_root: str, parser, splits: Optional[Dict[str, str]]):
        """
        Given downloaded and unzipped data of a particular data format,
        parse and prepare in pybrat style for use.

        To access the data, the name of each split provided in the dictionary is instantiated with a List[Examples]; thus if your `splits` kwarg keys are "train", "dev", "test", then the customdataset will have `train`, `dev`, `test` attributes.

        Ensure that the split names match expectations for downstream applications.

        :param data_root: Head directory of your data files
        :param parser: The function/parser that can process your data type and provide in the pybrat internal structure
        :param splits: Name of each data split (keys) and folder names
        """

        logger.info(f"Number of datasplits= {len(self.split_names)}")

        for split, fname in self.split_names.items():
            dtype_path = os.path.join(self.data_root, fname)
            setattr(self, split, parser.parse(dtype_path))
