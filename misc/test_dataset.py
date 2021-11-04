"""
Unittests to test each dataset example
"""

import unittest
from dataset import Dataset


class TestChemProt(unittest.TestCase):
    """Tests ChemProt dataset loaded properly"""

    def test_load(self):

        data_root = "/home/natasha/Projects/huggingface_bionlp/biomedical/datasets/ChemProt_Corpus"

        _TRAIN = "chemprot_training/brat"
        _DEV = "chemprot_development/brat"
        _TEST = "chemprot_test_gs/brat"
        _SAMPLE = "chemprot_sample/brat"

        splits = {"train": _TRAIN, "dev": _DEV, "test": _TEST, "sample": _SAMPLE}

        fmt = "brat"

        dataset = Dataset(data_root, splits, fmt)


if __name__ == "__main__":
    unittest.main()
