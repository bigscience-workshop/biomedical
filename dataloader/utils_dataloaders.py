from pathlib import Path

from flair.datasets import biomedical as hunflair
from pybrat.parser import BratParser


DATA_PATH = Path(__file__).parent / "data"
CACHE_PATH = DATA_PATH / "tmp"


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


class JNLPBA:
    """
    Original corpus of the JNLPBA shared task.
    For further information see Kim et al.:
      Introduction to the Bio-Entity Recognition Task at JNLPBA
      https://www.aclweb.org/anthology/W04-1213.pdf
    """

    train_dir = DATA_PATH / "JNLPBA" / "train"
    test_dir = DATA_PATH / "JNLPBA" / "test"

    def __init__(self):
        if not list(self.train_dir.glob("*")) or not list(self.test_dir.glob("*")):
            self._download()
        parser = BratParser(error="ignore")
        self.train_documents = parser.parse(self.train_dir)
        self.test_documents = parser.parse(self.test_dir)

    @classmethod
    def _download(cls):
        train_dataset = hunflair.HunerJNLPBA.download_and_prepare_train(
            data_folder=CACHE_PATH / "JNLPBA", sentence_tag=" "
        )
        test_dataset = hunflair.HunerJNLPBA.download_and_prepare_test(
            data_folder=CACHE_PATH / "JNLPBA", sentence_tag=" "
        )
        _write_hunflair_internal_to_brat(train_dataset, cls.train_dir)
        _write_hunflair_internal_to_brat(test_dataset, cls.test_dir)


class CellFinder:
    """
    Original CellFinder corpus containing cell line, species and gene annotations.
    For futher information see Neves et al.:
        Annotating and evaluating text for stem cell research
        https://pdfs.semanticscholar.org/38e3/75aeeeb1937d03c3c80128a70d8e7a74441f.pdf
    """

    train_dir = DATA_PATH / "cellfinder" / "train"

    def __init__(self):
        if not list(self.train_dir.glob("*")):
            self._download()
        parser = BratParser(error="ignore")
        self.train_documents = parser.parse(self.train_dir)

    @classmethod
    def _download(cls):
        train_dataset = hunflair.CELL_FINDER.download_and_prepare(
            data_folder=CACHE_PATH / "cellfinder"
        )
        _write_hunflair_internal_to_brat(train_dataset, cls.train_dir)


class Linneaus:
    """
    Original LINNEAUS corpus containing species annotations.
    For further information see Gerner et al.:
         LINNAEUS: a species name identification system for biomedical literature
         https://www.ncbi.nlm.nih.gov/pubmed/20149233
    """

    train_dir = DATA_PATH / "linneaus" / "train"

    def __init__(self):
        if not list(self.train_dir.glob("*")):
            self._download()
        parser = BratParser(error="ignore")
        self.train_documents = parser.parse(self.train_dir)

    @classmethod
    def _download(cls):
        train_dataset = hunflair.LINNEAUS.download_and_parse_dataset(
            data_dir=CACHE_PATH / "linneaus"
        )
        _write_hunflair_internal_to_brat(train_dataset, cls.train_dir)