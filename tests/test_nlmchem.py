#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test NLMChem dataloader
"""

import unittest

from datasets import load_dataset


class TestKBFeatures(unittest.TestCase):
    """
    Tests for features in dataset
    """

    def setUp(self):
        self.dataset = load_dataset("examples/nlmchem.py", "bigbio")

    def test_source(self):
        """
        Verify that the source view is present
        """

        dataset = load_dataset("examples/nlmchem.py", "source")

        self.assertIsNotNone(dataset)

    def test_passages_offsets(self):
        """
        Verify that the passages offsets are correct, i.e.: passage text == text extracted via the passage offsets
        """

        for split in self.dataset:

            for example in self.dataset[split]:

                example_text = " ".join([p["text"] for p in example["passages"]])

                for p in example["passages"]:
                    start, end = p["offsets"][0]
                    self.assertEqual(example_text[start:end], p["text"])

    def test_entities_offsets(self):
        """
        Verify that the entities offsets are correct, i.e.: entity text == text extracted via the entity offsets
        """
        for split in self.dataset:

            for example in self.dataset[split]:

                example_text = " ".join([p["text"] for p in example["passages"]])

                for entity in example["entities"]:
                    for idx, (start, end) in enumerate(entity["offsets"]):
                        self.assertEqual(example_text[start:end], entity["text"][idx])


if __name__ == "__main__":
    unittest.main()
