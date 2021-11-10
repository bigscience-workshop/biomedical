"""
TODO: Ask Leon about cache_path in Linneaus
"""
import os
from pathlib import Path
from typing import List, Optional, Dict

from flair.datasets import biomedical as hunflair
from pybrat.parser import BratParser

from .downloaders import download_data, uncompress_data, chemprot_2_standoff
from .formatters import BioCXML


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
            f.write(text)


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
