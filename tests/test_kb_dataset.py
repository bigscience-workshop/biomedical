import argparse
import sys
import unittest
from collections import defaultdict
from difflib import ndiff
from pathlib import Path
from pprint import pformat

sys.path.append(str(Path(__file__).parent.parent))

import datasets
from schemas.kb import features



class TestKBDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset_source = datasets.load_dataset(sys.argv[1], name="source")
        self.dataset_bigbio = datasets.load_dataset(sys.argv[1], name="bigbio")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datset_script")
    args = parser.parse_args()

    unittest.TextTestRunner().run(TestKBDataset())
