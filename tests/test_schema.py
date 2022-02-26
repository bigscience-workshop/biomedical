"""
Test whether proper schema keys are allocated for data-loader.
This only checks 1 example because the output of generate_examples is a generator
"""
import os
import glob
import unittest
import importlib
import datasets
import inspect
import logging as log
from datasets import load_dataset
from .key_constants import (
    KBSchema,
    QASchema,
    EntailmentSchema,
    Text2TextSchema,
    TextSchema,
    PairsSchema,
)

log.basicConfig(level=log.INFO)


class TestSchema(unittest.TestCase):
    """
    Given an task in `datasets`, checks to see whether
    To work, this should be run
    """

    def check_name(self):
        """
        Check if the directory structure is correct
        """
        dataset = [
            i
            for i in glob.glob("biomeddatasets/*/*")
            if ".py" in i and "__pycache" not in i
        ][0]

        dataset_module = dataset.replace("/", ".").strip(".py")
        dataset_name = dataset_module.split(".")
        self.assertTrue(
            dataset_name[-2] == dataset_name[-1],
            "Dataloader does not match directory name",
        )

    def test_schema(self):
        """Search supported tasks within a dataset"""

        dataset = [
            i
            for i in glob.glob("biomeddatasets/*/*")
            if ".py" in i and "__pycache" not in i
        ][0]

        dataset_module = dataset.replace("/", ".").strip(".py")
        dataset_name = dataset_module.split(".")[-1]

        # Import the _SUPPORTED_TASKS
        _SUPPORTED_TASKS = importlib.import_module(
            dataset_module
        )._SUPPORTED_TASKS

        data = load_dataset(dataset, name="bigbio")

        for task in _SUPPORTED_TASKS:

            if task == "kb":
                log.info("NER/NED/RE/Event Extraction/Coref Task")

                schema = KBSchema()

                for split in data.keys():
                    example = data[split][0]

                    # Check for mandatory keys
                    mandatory_keys = all(
                        [key in example for key in schema.keys]
                    )
                    self.assertTrue(
                        mandatory_keys,
                        "id/document_id/passages missing from bigbio view",
                    )

                    # Check for optional keys (ent/reln/coref/event)
                    present_opt_keys = [
                        key for key in schema.opt_keys if key in example
                    ]

                    self.assertTrue(
                        len(present_opt_keys) > 0,
                        "at least one key: entities/events/relations/coreferences not present",
                    )

                    # For defined keys, check their details are correct
                    for key in present_opt_keys:
                        attr_keys = getattr(schema, key, None)
                        self.assertTrue(
                            any([k in example for k in attr_keys])
                        )

            elif task == "qa":
                log.info("Question-Answering Task")

                schema = QASchema()

                for split in data.keys():
                    example = data[split][0]

                    # Check for mandatory keys
                    mandatory_keys = all(
                        [key in example for key in schema.keys]
                    )
                    self.assertTrue(
                        mandatory_keys,
                        "/".join(schema.keys) + "missing from bigbio view",
                    )

            elif task == "entailment":
                log.info("Entailment Task")

                schema = EntailmentSchema()

                for split in data.keys():
                    example = data[split][0]

                    # Check for mandatory keys
                    mandatory_keys = all(
                        [key in example for key in schema.keys]
                    )
                    self.assertTrue(
                        mandatory_keys,
                        "/".join(schema.keys) + "missing from bigbio view",
                    )

            elif task == "text_to_text":
                log.info("Translation/Summarization/Paraphrasing Task")

                schema = Text2TextSchema()

                for split in data.keys():
                    example = data[split][0]

                    # Check for mandatory keys
                    mandatory_keys = all(
                        [key in example for key in schema.keys]
                    )
                    self.assertTrue(
                        mandatory_keys,
                        "/".join(schema.keys) + "missing from bigbio view",
                    )

            elif task == "text":
                log.info("Sentence/Phrase/Text Classification Task")

                schema = TextSchema()

                for split in data.keys():
                    example = data[split][0]

                    # Check for mandatory keys
                    mandatory_keys = all(
                        [key in example for key in schema.keys]
                    )
                    self.assertTrue(
                        mandatory_keys,
                        "/".join(schema.keys) + "missing from bigbio view",
                    )

            elif task == "pairs":
                log.info("Pair Labels Task")

                schema = PairsSchema()

                for split in data.keys():
                    example = data[split][0]

                    # Check for mandatory keys
                    mandatory_keys = all(
                        [key in example for key in schema.keys]
                    )
                    self.assertTrue(
                        mandatory_keys,
                        "/".join(schema.keys) + "missing from bigbio view",
                    )

            else:
                raise ValueError(
                    task
                    + " not specified; only 'kb', 'qa', 'entailment', 'text_to_text', 'text', 'pairs' allowed"
                )
