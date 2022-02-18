#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for KB template
"""

import argparse
import sys
import unittest
from typing import Iterator, Optional, Union

from datasets import load_dataset

OFFSET_ERROR_MSG = (
    "\n"
    "There are features with wrong offsets!"
    "This is not a hard failure, as it is common for this type of datasets"
    "However, if the error list is long (e.g. >10) you should double check your code"
)


def _get_example_text(example: dict) -> str:
    """
    Get full text example
    """
    return " ".join([t for p in example["passages"] for t in p["text"]])


def _check_offsets(
    example_text: str,
    offsets: list[list[int]],
    texts: list[str],
) -> Iterator:

    for idx, (start, end) in enumerate(offsets):

        try:
            text = texts[idx]

        # TODO: riase error?
        except IndexError:
            print("Pairing of a list of offsets with a single text is not allowed!")

        by_offset_text = example_text[start:end]

        if by_offset_text != text:
            yield f" text:`{text}` != offsets_text:`{by_offset_text}`"


class TestKBFeatures(unittest.TestCase):
    """
    Tests for features in dataset
    """

    PATH: str
    DATA_DIR: str
    USE_AUTH_TOKEN: Optional[Union[bool, str]]

    def setUp(self):
        self.dataset = load_dataset(
            self.PATH,
            name="bigbio",
            data_dir=self.DATA_DIR,
            use_auth_token=self.USE_AUTH_TOKEN,
        )

    def test_passages_offsets(self):
        """
        Verify that the passages offsets are correct,
        i.e.: passage text == text extracted via the passage offsets
        """

        for split in self.dataset:

            if "passages" in self.dataset[split].features:

                for example in self.dataset[split]:

                    example_text = _get_example_text(example)

                    for entity in example["passages"]:
                        for idx, (start, end) in enumerate(entity["offsets"]):
                            self.assertEqual(
                                example_text[start:end], entity["text"][idx]
                            )

    def test_entities_offsets(self):
        """
        Verify that the entities offsets are correct,
        i.e.: entity text == text extracted via the entity offsets
        """

        errors = []

        for split in self.dataset:

            if "entities" in self.dataset[split].features:

                for example in self.dataset[split]:

                    example_id = example["id"]
                    example_text = _get_example_text(example)

                    for entity in example["entities"]:

                        for msg in _check_offsets(
                            example_text=example_text,
                            offsets=entity["offsets"],
                            texts=entity["text"],
                        ):

                            entity_id = entity["id"]
                            errors.append(
                                f"Example:{example_id} - entity:{entity_id} " + msg
                            )

        if len(errors) > 0:
            self.fail(msg="\n".join(errors) + OFFSET_ERROR_MSG)

    # UNTESTED: no dataset example
    def test_events_offsets(self):
        """
        Verify that the events' trigger offsets are correct,
        i.e.: trigger text == text extracted via the trigger offsets
        """

        errors = []

        for split in self.dataset:

            if "events" in self.dataset[split].features:

                for example in self.dataset[split]:

                    example_id = example["id"]
                    example_text = _get_example_text(example)

                    for event in example["events"]:

                        for msg in _check_offsets(
                            example_text=example_text,
                            offsets=event["trigger"]["offsets"],
                            texts=event["trigger"]["text"],
                        ):

                            event_id = event["id"]
                            errors.append(
                                f"Example:{example_id} - event:{event_id} " + msg
                            )

        if len(errors) > 0:
            self.fail(msg="\n".join(errors) + OFFSET_ERROR_MSG)

    def test_coref_ids(self):
        """
        Verify that coreferences ids are entities

        from `examples/test_n2c2_2011_coref.py`
        """

        for split in self.dataset:

            if "coreferences" in self.dataset[split].features:

                for example in self.dataset[split]:
                    entity_lookup = {ent["id"]: ent for ent in example["entities"]}

                    # check all coref entity ids are in entity lookup
                    for coref in example["coreferences"]:
                        for entity_id in coref["entity_ids"]:
                            assert entity_id in entity_lookup


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unit tests for dataset with KB schema. Args are passed to `datasets.load_dataset`"
    )

    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--use_auth_token", default=None)

    args = parser.parse_args()

    TestKBFeatures.PATH = args.path
    TestKBFeatures.DATA_DIR = args.data_dir
    TestKBFeatures.USE_AUTH_TOKEN = args.use_auth_token

    sys.argv = sys.argv[:1]

    unittest.main()
