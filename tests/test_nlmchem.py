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
        self.dataset = load_dataset("examples/nlmchem.py")

    def test_offsets(self):
        """
        Verify that the entities offsets are correct.
        """
        for split in self.dataset:

            for example in self.dataset[split]:

                example_text = " ".join(example["passages"]["text"])

                entities_text = example["entities"]["text"]

                entities_offsets = example["entities"]["offsets"]

                for idx, offsets in enumerate(entities_offsets):
                    for (start, end) in offsets:
                        self.assertEqual(
                            example_text[start:end], entities_text[idx][0][0]
                        )


if __name__ == "__main__":
    unittest.main()
