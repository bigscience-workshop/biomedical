import argparse
import sys
import unittest
from collections import defaultdict
from collections.abc import Iterable
from difflib import ndiff
from pathlib import Path
from pprint import pformat
from typing import Iterator, Optional, Union

sys.path.append(str(Path(__file__).parent.parent))

import datasets

from schemas.kb import features

OFFSET_ERROR_MSG = (
    "\n\n"
    "There are features with wrong offsets!"
    " This is not a hard failure, as it is common for this type of datasets."
    " However, if the error list is long (e.g. >10) you should double check your code."
)


def _get_example_text(example: dict) -> str:
    """
    Get full text example
    """
    return " ".join([t for p in example["passages"] for t in p["text"]])


class TestKBDataset(unittest.TestCase):

    PATH: str
    NAME: str
    DATA_DIR: str
    USE_AUTH_TOKEN: Optional[Union[bool, str]]

    def setUp(self) -> None:

        self.dataset_source = datasets.load_dataset(
            self.PATH,
            name="source",
            data_dir=self.DATA_DIR,
            use_auth_token=self.USE_AUTH_TOKEN,
        )

        self.dataset_bigbio = datasets.load_dataset(
            self.PATH,
            name=self.NAME,
            data_dir=self.DATA_DIR,
            use_auth_token=self.USE_AUTH_TOKEN,
        )
        # self.dataset_source = datasets.load_dataset(sys.argv[1], name="source")
        # self.dataset_bigbio = datasets.load_dataset(sys.argv[1], name="bigbio")

    def print_statistics(self):
        for split_name, split in self.dataset_bigbio.items():
            print(split_name)
            print("=" * 10)

            counter = defaultdict(int)
            for example in split:
                for feature_name, feature in features.items():
                    if isinstance(feature, datasets.Value) or isinstance(
                        feature_name, dict
                    ):
                        if example[feature_name]:
                            counter[feature_name] += 1
                    else:
                        counter[feature_name] += len(example[feature_name])

            for k, v in counter.items():
                print(f"{k}: {v}")
            print()

    def runTest(self):
        self.print_statistics()
        self.test_is_bigbio_schema_compatible()
        self.test_are_ids_globally_unique()
        self.test_do_all_referenced_ids_exist()
        self.test_passages_offsets()
        self.test_entities_offsets()
        self.test_events_offsets()
        self.test_coref_ids()

    def test_is_bigbio_schema_compatible(self):
        for split in self.dataset_bigbio.values():
            if not split.features == features:
                s1 = pformat(split.features).splitlines()
                s2 = pformat(features).splitlines()
                print("\n".join(ndiff(s1, s2)))
                assert split.features == features

    def test_are_ids_globally_unique(self):
        for split in self.dataset_bigbio.values():
            ids_seen = set()
            for example in split:
                self._assert_ids_globally_unique(example, ids_seen=ids_seen)

    def test_do_all_referenced_ids_exist(self):
        for split in self.dataset_bigbio.values():
            for example in split:
                referenced_ids = set()
                existing_ids = set()

                referenced_ids.update(self._get_referenced_ids(example))
                existing_ids.update(self._get_existing_referable_ids(example))

                for ref_id, ref_type in referenced_ids:
                    if ref_type == "event_arg":
                        self.assertTrue(
                            (ref_id, "entity") in existing_ids
                            or (ref_id, "event") in existing_ids
                        )
                    else:
                        self.assertIn((ref_id, ref_type), existing_ids)

    def _assert_ids_globally_unique(
        self, collection, ids_seen: set, ignore_assertion: bool = False
    ):
        if isinstance(collection, dict):
            for k, v in collection.items():
                if isinstance(v, dict):
                    self._assert_ids_globally_unique(v, ids_seen)
                elif isinstance(v, list):
                    for elem in v:
                        self._assert_ids_globally_unique(elem, ids_seen)
                else:
                    if k == "id":
                        if not ignore_assertion:
                            self.assertNotIn(v, ids_seen)
                        ids_seen.add(v)
        elif isinstance(collection, list):
            for elem in collection:
                self._assert_ids_globally_unique(elem, ids_seen)

    def _get_referenced_ids(self, example):
        referenced_ids = []

        for event in example["events"]:
            for argument in event["arguments"]:
                referenced_ids.append((argument["ref_id"], "event_arg"))

        for coreference in example["coreferences"]:
            for entity_id in coreference["entity_ids"]:
                referenced_ids.append((entity_id, "entity"))

        for relation in example["relations"]:
            referenced_ids.append((relation["arg1_id"], "entity"))
            referenced_ids.append((relation["arg2_id"], "entity"))

        return referenced_ids

    def _get_existing_referable_ids(self, example):
        existing_ids = []

        for entity in example["entities"]:
            existing_ids.append((entity["id"], "entity"))

        for event in example["events"]:
            existing_ids.append((event["id"], "event"))

        return existing_ids

    def test_passages_offsets(self):
        """
        Verify that the passages offsets are correct,
        i.e.: passage text == text extracted via the passage offsets
        """

        for split in self.dataset_bigbio:

            if "passages" in self.dataset_bigbio[split].features:

                for example in self.dataset_bigbio[split]:

                    example_text = _get_example_text(example)

                    for passage in example["passages"]:

                        text = passage["text"]
                        offsets = passage["offsets"]

                        with self.subTest(
                            "Offsets in passages must have only one element",
                            offsets=offsets,
                        ):
                            self.assertEqual(len(offsets), 1)

                        with self.subTest(
                            "Text in passages must be an Iterable",
                            text=text,
                        ):
                            self.assertIsInstance(text, Iterable)

                        with self.subTest(
                            "Text in passages must have only one element",
                            text=text,
                        ):
                            self.assertEqual(len(text), 1)

                        for idx, (start, end) in enumerate(offsets):
                            self.assertEqual(example_text[start:end], text[idx])

    def _check_offsets(
        self,
        example_text: str,
        offsets: list[list[int]],
        texts: list[str],
    ) -> Iterator:

        with self.subTest(
            "# of texts must be equal to # of offsets", texts=texts, offsets=offsets
        ):
            self.assertEqual(len(texts), len(offsets))

        with self.subTest(
            "Text fields paired with offsets must be in the form [`text`, ...]",
            texts=texts,
        ):
            self.assertIsInstance(texts, Iterable)

        with self.subTest(
            "All offsets must be in the form [(lo1, hi1), ...]", offsets=offsets
        ):
            self.assertTrue(all(len(o) == 2 for o in offsets))

        # offsets are always list of lists
        for idx, (start, end) in enumerate(offsets):

            by_offset_text = example_text[start:end]
            text = texts[idx]

            if by_offset_text != text:
                yield f" text:`{text}` != text_by_offset:`{by_offset_text}`"

    def test_entities_offsets(self):
        """
        Verify that the entities offsets are correct,
        i.e.: entity text == text extracted via the entity offsets
        """

        errors = []

        for split in self.dataset_bigbio:

            if "entities" in self.dataset_bigbio[split].features:

                for example in self.dataset_bigbio[split]:

                    example_id = example["id"]
                    example_text = _get_example_text(example)

                    for entity in example["entities"]:

                        for msg in self._check_offsets(
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

        for split in self.dataset_bigbio:

            if "events" in self.dataset_bigbio[split].features:

                for example in self.dataset_bigbio[split]:

                    example_id = example["id"]
                    example_text = _get_example_text(example)

                    for event in example["events"]:

                        for msg in self._check_offsets(
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

        for split in self.dataset_bigbio:

            if "coreferences" in self.dataset_bigbio[split].features:

                for example in self.dataset_bigbio[split]:
                    entity_lookup = {ent["id"]: ent for ent in example["entities"]}

                    # check all coref entity ids are in entity lookup
                    for coref in example["coreferences"]:
                        for entity_id in coref["entity_ids"]:
                            assert entity_id in entity_lookup


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("datset_script")
    # args = parser.parse_args()
    parser = argparse.ArgumentParser(
        description="Unit tests for dataset with KB schema. Args are passed to `datasets.load_dataset`"
    )

    parser.add_argument("--path", type=str, required=True)
    parser.add_argument(
        "--name",
        type=str,
        default="bigbio",
        help="For datasets supporting multiple tasks, e.g. `bigbio-translation`",
    )
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--use_auth_token", default=None)

    args = parser.parse_args()

    TestKBDataset.PATH = args.path
    TestKBDataset.NAME = args.name
    TestKBDataset.DATA_DIR = args.data_dir
    TestKBDataset.USE_AUTH_TOKEN = args.use_auth_token

    unittest.TextTestRunner().run(TestKBDataset())
