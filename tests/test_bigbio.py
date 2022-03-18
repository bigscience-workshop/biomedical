"""
Unit-tests to ensure tasks adhere to big-bio schema.
"""
import sys
import warnings

import datasets
from datasets import load_dataset, Features
from schemas import (
    kb_features,
    qa_features,
    entailment_features,
    text2text_features,
    text_features,
    pairs_features,
)

import unittest
import importlib
import argparse
from collections import defaultdict
from difflib import ndiff
from pathlib import Path
from pprint import pformat
from typing import Iterator, Optional, Union, Iterable, List

sys.path.append(str(Path(__file__).parent.parent))

import logging


logger = logging.getLogger(__name__)


_TASK_TO_SCHEMA = {
    "NER": "KB",
    "NED": "KB",
    "EE": "KB",
    "RE": "KB",
    "COREF": "KB",
    "QA": "QA",
    "TE": "TE",
    "STS": "PAIRS",
    "PARA": "T2T",
    "TRANSL": "T2T",
    "SUM": "T2T",
    "TXTCLASS": "TEXT",
}

_VALID_TASKS = set(_TASK_TO_SCHEMA.keys())
_VALID_SCHEMA = set(_TASK_TO_SCHEMA.values())

_SCHEMA_TO_FEAUTURES = {
    "KB": kb_features,
    "QA": qa_features,
    "TE": entailment_features,
    "T2T": text2text_features,
    "TEXT": text_features,
    "PAIRS": pairs_features,
}


def _get_example_text(example: dict) -> str:
    """
    Concatenate all text from passages in an example of a KB schema
    :param example: An instance of the KB schema
    """
    return " ".join([t for p in example["passages"] for t in p["text"]])


OFFSET_ERROR_MSG = (
    "\n\n"
    "There are features with wrong offsets!"
    " This is not a hard failure, as it is common for this type of datasets."
    " However, if the error list is long (e.g. >10) you should double check your code."
)


class TestDataLoader(unittest.TestCase):
    """
    Given a dataset script that has been implemented, check if it adheres to the `bigbio` schema.

    The test
    """

    PATH: str
    NAME: str
    DATA_DIR: Optional[str]
    USE_AUTH_TOKEN: Optional[Union[bool, str]]

    def runTest(self):
        """
         Run all tests that check:
         (1) test_name: Checks if dataloading script has correct path format
         (2) setUp: Checks data and _SUPPORTED_TASKS can be loaded
         (3) print_statistics: Counts number of all possible schema keys/instances of the examples
         (4) test_schema: confirms big-bio keys present
         (5) test_are_ids_globally_unique: Checks if all examples have a unique identifier

         # KB-Specific tests
         (6) test_do_all_referenced_ids_exist: Check if any sub-key (ex: entities/events etc.) have referenced keys
         (7) test_passages_offsets: Check if text matches offsets in passages
         (8) test_entities_offsets: Check if text matches offsets in entities
         (9) test_events_offsets: Check if text matches offsets in events
        (10) test_coref_ids: Check if text matches offsets in coreferences

        """  # noqa
        self.test_name()
        self.setUp()
        self.test_are_ids_globally_unique()

        # KB-specific unit-tests
        for task in self._SUPPORTED_TASKS:
            self.test_schema(task)

            mapped_features = _SCHEMA_TO_FEAUTURES[_TASK_TO_SCHEMA[task]]
            self.print_statistics(mapped_features)

            if _TASK_TO_SCHEMA[task] == "KB":
                self.test_do_all_referenced_ids_exist()
                self.test_passages_offsets()
                self.test_entities_offsets()
                self.test_events_offsets()
                self.test_coref_ids()

    def setUp(self) -> None:
        """Load original and big-bio schema views"""

        logger.info(f"self.PATH: {self.PATH}")
        logger.info(f"self.NAME: {self.NAME}")
        logger.info(f"self.SCHEMA: {self.SCHEMA}")
        logger.info(f"self.DATA_DIR: {self.DATA_DIR}")

        # Get task type of the dataset
        logger.info("Checking for _SUPPORTED_TASKS ...")
        module = self.PATH
        if module.endswith(".py"):
            module = module[:-3]
        module = module.replace("/", ".")
        self._SUPPORTED_TASKS = importlib.import_module(module)._SUPPORTED_TASKS
        logger.info(f"Found _SUPPORTED_TASKS={self._SUPPORTED_TASKS}")

        invalid_tasks = set(self._SUPPORTED_TASKS) - _VALID_TASKS
        if len(invalid_tasks) > 0:
            raise ValueError(f"Found invalid supported tasks {invalid_tasks}. Must be one of {_VALID_TASKS}")

        config_name = f"{self.NAME}_source"
        logger.info(f"Checking load_dataset with config name {config_name}")
        self.dataset_source = datasets.load_dataset(
            self.PATH,
            name=config_name,
            data_dir=self.DATA_DIR,
            use_auth_token=self.USE_AUTH_TOKEN,
        )

        config_name = f"{self.NAME}_bigbio_{self.SCHEMA.lower()}"
        logger.info(f"Checking load_dataset with config name {config_name}")
        self.dataset_bigbio = datasets.load_dataset(
            self.PATH,
            name=config_name,
            data_dir=self.DATA_DIR,
            use_auth_token=self.USE_AUTH_TOKEN,
        )

    def print_statistics(self, schema: Features):
        """
        Gets sample statistics, for each split and sample of the number of
        features in the schema present; only works for the big-bio schema.

        :param schema_type: Type of schema to reference features from
        """  # noqa
        logger.info("Gathering schema statistics")
        for split_name, split in self.dataset_bigbio.items():
            print(split_name)
            print("=" * 10)

            counter = defaultdict(int)
            for example in split:
                for feature_name, feature in schema.items():
                    if example.get(feature_name, None) is not None:
                        if isinstance(feature, datasets.Value):
                            if example[feature_name]:
                                counter[feature_name] += 1
                        else:
                            counter[feature_name] += len(
                                example[feature_name]
                            )

            for k, v in counter.items():
                print(f"{k}: {v}")
            print()

    def test_are_ids_globally_unique(self):
        """
        Tests each example in a split has a unique ID.
        """
        logger.info("Checking global ID uniqueness")
        for split in self.dataset_bigbio.values():
            ids_seen = set()
            for example in split:
                self._assert_ids_globally_unique(example, ids_seen=ids_seen)

    def test_do_all_referenced_ids_exist(self):
        """
        Checks if referenced IDs are correctly labeled.
        """
        logger.info("Checking if referenced IDs are properly mapped")
        for split in self.dataset_bigbio.values():
            for example in split:
                referenced_ids = set()
                existing_ids = set()

                referenced_ids.update(self._get_referenced_ids(example))
                existing_ids.update(
                    self._get_existing_referable_ids(example)
                )

                for ref_id, ref_type in referenced_ids:
                    if ref_type == "event_arg":
                        if not (
                            (ref_id, "entity") in existing_ids
                            or (ref_id, "event") in existing_ids
                        ):
                            logger.warning(f"Referenced element ({ref_id}, entity/event) could not be found in existing ids {existing_ids}. Please make sure that this is not because of a bug in your data loader.")
                    else:
                        logger.warning(f"Referenced element {(ref_id, ref_type)} could not be found in existing ids {existing_ids}. Please make sure that this is not because of a bug in your data loader.")

    def test_passages_offsets(self):
        """
        Verify that the passages offsets are correct,
        i.e.: passage text == text extracted via the passage offsets
        """  # noqa
        logger.info("KB ONLY: Checking passage offsets")
        for split in self.dataset_bigbio:

            if "passages" in self.dataset_bigbio[split].features:

                for example in self.dataset_bigbio[split]:

                    example_text = _get_example_text(example)

                    for passage in example["passages"]:

                        text = passage["text"]
                        offsets = passage["offsets"]

                        self._test_is_list(
                            msg="Text in passages must be a list", field=text
                        )

                        self._test_is_list(
                            msg="Offsets in passages must be a list",
                            field=offsets,
                        )

                        self._test_has_only_one_item(
                            msg="Offsets in passages must have only one element",
                            field=offsets,
                        )

                        self._test_has_only_one_item(
                            msg="Text in passages must have only one element",
                            field=text,
                        )

                        for idx, (start, end) in enumerate(offsets):
                            msg = f" text:`{example_text[start:end]}` != text_by_offset:`{text[idx]}`"
                            self.assertEqual(
                                example_text[start:end], text[idx], msg
                            )

    def test_entities_offsets(self):
        """
        Verify that the entities offsets are correct,
        i.e.: entity text == text extracted via the entity offsets
        """  # noqa
        logger.info("KB ONLY: Checking entity offsets")
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
                                f"Example:{example_id} - entity:{entity_id} "
                                + msg
                            )

        if len(errors) > 0:
            logger.warning(msg="\n".join(errors) + OFFSET_ERROR_MSG)

    # UNTESTED: no dataset example
    def test_events_offsets(self):
        """
        Verify that the events' trigger offsets are correct,
        i.e.: trigger text == text extracted via the trigger offsets
        """
        logger.info("KB ONLY: Checking event offsets")
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
                                f"Example:{example_id} - event:{event_id} "
                                + msg
                            )

        logger.warning(msg="\n".join(errors) + OFFSET_ERROR_MSG)

    def test_coref_ids(self):
        """
        Verify that coreferences ids are entities

        from `examples/test_n2c2_2011_coref.py`
        """  # noqa
        logger.info("KB ONLY: Checking coref offsets")
        for split in self.dataset_bigbio:

            if "coreferences" in self.dataset_bigbio[split].features:

                for example in self.dataset_bigbio[split]:
                    entity_lookup = {
                        ent["id"]: ent for ent in example["entities"]
                    }

                    # check all coref entity ids are in entity lookup
                    for coref in example["coreferences"]:
                        for entity_id in coref["entity_ids"]:
                            assert entity_id in entity_lookup

    def test_name(self):
        """
        Checks if the dataloader script is in the right nested structure.
        All dataloader scripts must be of the following format:

        biomedical/biomeddatasets/<name_of_dataset>/<name_of_dataloader_script>
        """  # noqa
        datafolder = Path("examples")
        datascript = (datafolder / self.NAME).with_suffix(".py")
        self.assertTrue(datafolder.exists(), "Folder not named " + self.NAME)

        self.assertTrue(
            datascript.exists(),
            "Script not named " + self.NAME + ".py",
        )

    def test_schema(self, task: str):
        """Search supported tasks within a dataset and verify big-bio schema"""

        mapped_schema = _TASK_TO_SCHEMA[task]
        logger.info(f"task={task}, mapped_schema={mapped_schema}")

        if mapped_schema == "KB":

            features = kb_features
            logger.info(f"all feature names: {set(features.keys())}")

            # construct task specific required keys
            opt_keys = [
                "entities",
                "events",
                "coreferences",
                "relations",
            ]

            needed_keys = [key for key in features.keys() if key not in opt_keys]

            if task == "NER":
                sub_keys = ["entities"]
            elif task == "RE":
                sub_keys = ["entities", "relations"]
            elif task == "NED":
                sub_keys = ["entities"]
            elif task == "COREF":
                sub_keys = ["entities", "coreferences"]
            elif task == "EE":
                sub_keys = ["events"]
            else:
                raise ValueError(f"Task {task} not recognized")


            logger.info(f"needed_keys: {needed_keys}")
            logger.info(f"sub_keys: {sub_keys}")


            for split in self.dataset_bigbio.keys():
                example = self.dataset_bigbio[split][0]

                # Check for mandatory keys
                missing_keys = set(needed_keys) - set(example.keys())
                self.assertTrue(
                    len(missing_keys) == 0,
                    f"{missing_keys} are missing from bigbio view",
                )

                for key in sub_keys:
                    for attrs in features[key]:
                        self.assertTrue(self._check_subkey(example[key][0], attrs))

                # miscellaneous keys not affiliated with a type (ex: NER dataset with events)
                extra_keys = set(example.keys()) - set(needed_keys) - set(sub_keys)
                logger.info(f"extra_keys in {split}: {extra_keys}")
                for key in extra_keys:
                    if key in features.keys():
                        for attrs in features[key]:
                            self.assertTrue(self._check_subkey(example[key][0], attrs))

        elif mapped_schema == "QA":
            logger.info("Question-Answering Schema")
            self._check_keys(_SCHEMA_TO_FEAUTURES[mapped_schema], mapped_schema)

        elif mapped_schema == "TE":
            logger.info("Textual Entailment Schema")
            self._check_keys(_SCHEMA_TO_FEAUTURES[mapped_schema], mapped_schema)

        elif mapped_schema == "T2T":
            logger.info("Text to Text Schema")
            self._check_keys(_SCHEMA_TO_FEAUTURES[mapped_schema], mapped_schema)

        elif mapped_schema == "TEXT":
            logger.info("Text Schema")
            self._check_keys(_SCHEMA_TO_FEAUTURES[mapped_schema], mapped_schema)

        elif mapped_schema == "PAIRS":
            logger.info("Text Pair Schema")
            self._check_keys(_SCHEMA_TO_FEAUTURES[mapped_schema], mapped_schema)

        else:
            raise ValueError(
                f"{mapped_schema} not recognized. must be one of {set(_TASK_TO_SCHEMA.values())}"
            )

    @staticmethod
    def _check_subkey(inp, attrs):
        """Checks if subkeys (esp. in KB) have necessary criteria"""
        return all([k in inp for k in attrs.keys()])

    def _check_keys(self, schema: Features, schema_name: str):
        """Check if necessary keys are present in a given schema"""
        for split in self.dataset_bigbio.keys():
            example = self.dataset_bigbio[split][0]

            # Check for mandatory keys
            mandatory_keys = all([key in example for key in schema.keys()])
            self.assertTrue(
                mandatory_keys,
                "/".join(schema.keys())
                + " keys missing from bigbio view for schema_type = "
                + schema_name,
            )

    def _assert_ids_globally_unique(
        self,
        collection: Iterable,
        ids_seen: set,
        ignore_assertion: bool = False,
    ):
        """
        Checks if all IDs are globally unique across a feature list.
        This looks recursively through elements of arrays to check if every referenced ID is unique.

        :param collection: An iterable of features that contain NLP info (ex: entities, events)
        :param ids_seen: Set of previously seen numerical IDs (empty by default)
        :param ignore_assertion: Whether to raise an error if id was already seen.
        """  # noqa
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

        if example.get("events", None) is not None:
            for event in example["events"]:
                for argument in event["arguments"]:
                    referenced_ids.append((argument["ref_id"], "event_arg"))

        if example.get("coreferences", None) is not None:
            for coreference in example["coreferences"]:
                for entity_id in coreference["entity_ids"]:
                    referenced_ids.append((entity_id, "entity"))

        if example.get("relations", None) is not None:
            for relation in example["relations"]:
                referenced_ids.append((relation["arg1_id"], "entity"))
                referenced_ids.append((relation["arg2_id"], "entity"))

        return referenced_ids

    def _get_existing_referable_ids(self, example):
        existing_ids = []

        for entity in example["entities"]:
            existing_ids.append((entity["id"], "entity"))

        if example.get("events", None) is not None:
            for event in example["events"]:
                existing_ids.append((event["id"], "event"))

        return existing_ids

    def _test_is_list(self, msg: str, field: list):
        with self.subTest(
            msg,
            field=field,
        ):
            self.assertIsInstance(field, list)

    def _test_has_only_one_item(self, msg: str, field: list):
        with self.subTest(
            msg,
            field=field,
        ):
            self.assertEqual(len(field), 1)

    def _check_offsets(
        self,
        example_text: str,
        offsets: List[List[int]],
        texts: List[str],
    ) -> Iterator:
        """

        :param example_text:
        :param offsets:
        :param texts:

        """  # noqa

        if len(texts) != len(offsets):
            logger.warning(f"Number of texts {len(texts)} != number of offsets {len(offsets)}. Please make sure that this error already exists in the original data and was not introduced in the data loader.")

        self._test_is_list(
            msg="Text fields paired with offsets must be in the form [`text`, ...]",
            field=texts,
        )

        with self.subTest(
            "All offsets must be in the form [(lo1, hi1), ...]",
            offsets=offsets,
        ):
            self.assertTrue(all(len(o) == 2 for o in offsets))

        # offsets are always list of lists
        for idx, (start, end) in enumerate(offsets):

            by_offset_text = example_text[start:end]
            try:
                text = texts[idx]
            except IndexError:
                text = ""

            if by_offset_text != text:
                yield f" text:`{text}` != text_by_offset:`{by_offset_text}`"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Unit tests for BigBio datasets. Args are passed to `datasets.load_dataset`"
    )

    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="path to dataloader script"
    )
    parser.add_argument(
        "--schema",
        type=str,
        required=True,
        choices=list(_VALID_SCHEMA),
        help="specific bigbio schema to test.",
    )
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--use_auth_token", default=None)

    args = parser.parse_args()

    name = args.path.split(".py")[0].split("/")[-1]

    TestDataLoader.PATH = args.path
    TestDataLoader.NAME = name
    TestDataLoader.SCHEMA = args.schema
    TestDataLoader.DATA_DIR = args.data_dir
    TestDataLoader.USE_AUTH_TOKEN = args.use_auth_token

    unittest.TextTestRunner().run(TestDataLoader())
