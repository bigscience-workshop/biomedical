"""
Unittests to test each dataset example

Update - chemprot + ddi working
"""
import os
import unittest
from dataset import Dataset

# Place your biomedical dataset path here if it doesn't correspond
dataset_path = "../datasets/"

class TestDataset(unittest.TestCase):
    """Tests ChemProt dataset loaded properly"""

    def test_chemprot(self):
        """ Test the Chemprot Dataset (brat) """
        data_root = os.path.join(dataset_path, "ChemProt_Corpus")

        _TRAIN = "chemprot_training/brat"
        _DEV = "chemprot_development/brat"
        _TEST = "chemprot_test_gs/brat"
        _SAMPLE = "chemprot_sample/brat"

        splits = {"train": _TRAIN, "dev": _DEV, "test": _TEST, "sample": _SAMPLE}

        fmt = "brat"

        dataset = Dataset(data_root, splits, fmt)

    def test_ddi(self):
        """ Test the DDI dataset (brat) """
        data_root = os.path.join(dataset_path, "DDICorpusBrat")

        _TRAIN = "Train"
        _TEST = "Test"

        splits = {"train": _TRAIN, "test": _TEST}

        fmt = "brat"

        dataset = Dataset(data_root, splits, fmt)

    def test_bc5cdr(self):
        """ Test bc5cdr formatting changer """
        #data_root = os.path.join(dataset_path, "CDR_Data", "CDR.Corpus.v010516")

        #_TRAIN = "CDR_TrainingSet.BioC.xml"
        #_DEV = "CDR_DevelopmentSet.BioC.xml"
        #_TEST = "CDR_TestSet.BioC.xml"

        #fmt = "bioc_xml"

        #dataset = Dataset(data_root, splits, fmt)


    def test_badfmt(self):
        """ Check if a bad format is permitted """

if __name__ == "__main__":
    unittest.main()
