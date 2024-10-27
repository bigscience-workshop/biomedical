"""

"""
from lm_eval.base import BioTask

_CITATION = """
Biosses Citation
"""

class CustomDatasetBase(BioTask):
    VERSION = 0
    DATASET_PATH = "/home/natasha/Tutorials/biomedical/bigbio/biodatasets/custom_dataset"
    DATASET_NAME = None
    SPLIT = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            return self.dataset["train"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]
          
class CustomPairs(CustomDatasetBase):
    DATASET_NAME = "custom_bigbio_pairs"
