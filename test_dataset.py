"""
Unittests to test each dataset example

Update - chemprot + ddi working
"""
import os
import unittest
from dataloader.dataset import BioDataset, dataloader_lookup

# Place your biomedical dataset path here if it doesn't correspond
dataset_path = "datasets/"


class TestDataset(unittest.TestCase):
    """Tests ChemProt dataset loaded properly"""

    def test_bc5c5(self):
        """
        Test BC5CDR processing
        """
        dataset = "bc5cdr"
        data_root = dataset_path

        dataset = BioDataset(dataset, data_root)

    def test_jnlpba(self):
        """
        Test JNLPBA dataset with minimal specifications (dataset + data_root name)
        """
        dataset = "jnlpba"
        data_root = dataset_path

        dataset = BioDataset(dataset, data_root)

    def test_cellfinder(self):
        """Test CellFinder dataset with only dataset name provided"""
        dataset = BioDataset("cellfinder")

    def test_linneaus(self):
        """Test Linnaeus dataset with new data location + all caps"""
        dataset = "LINNEAUS"
        data_root = "./mydata"
        dataset = BioDataset(dataset, data_root)

    def test_ddi(self):
        """Test DDI downloads + processes properly"""
        dataset = BioDataset("ddi", dataset_path)

    #def test_drugprot(self):
    #    """Test DrugProt"""
    #    dataset = Dataset("drugprot", dataset_path)

    def test_chemprot(self):
        """Test ChemProt (deprecated; newest version is DrugProt)"""
        dataset = BioDataset("chemprot", dataset_path)

    def test_hunflair(self):
        for dataset_name in dataloader_lookup:
            if dataset_name.startswith("hunflair_"):
                dataset = BioDataset(dataset_name, dataset_path)
                assert len(dataset.data.train) > 0
                assert len(dataset.data.dev) > 0
                assert len(dataset.data.test) > 0

if __name__ == "__main__":
    unittest.main()
