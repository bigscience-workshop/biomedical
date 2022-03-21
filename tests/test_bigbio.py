"""
Unit-tests to ensure tasks adhere to big-bio schema.
"""
import argparse
import importlib
import sys
import unittest
import warnings
from collections import defaultdict
from difflib import ndiff
from pathlib import Path
from pprint import pformat
from typing import Iterable, Iterator, List, Optional, Union

import datasets
from datasets import DatasetDict, Features, load_dataset

from schemas import (entailment_features, kb_features, pairs_features,
                     qa_features, text2text_features, text_features)

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
_VALID_SCHEMAS = set(_TASK_TO_SCHEMA.values())

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
    " However, if the error list is long (e.g. >10) you should double check your code. \n\n"
)


class TestDataLoader(unittest.TestCase):
    """
    Given a dataset script that has been implemented, check if it adheres to the `bigbio` schema.

    The test
    """

    PATH: str
    SCHEMA: str
    SUBSET_ID: str
    DATA_DIR: Optional[str]
    USE_AUTH_TOKEN: Optional[Union[bool, str]]

    def runTest(self):
        """
         Run all tests that check:
         (1) [removed - full path to script is now passed in] test_name: Checks if
             dataloading script has correct path format
         (2) setUp: Checks data and _SUPPORTED_TASKS can be loaded
         (3) print_statistics: Counts number of all possible schema keys/instances of the examples
         (4) test_schema: confirms big-bio keys present
         (5) test_are_ids_globally_unique: Checks if all examples have a unique identifier

         # KB-Specific tests
         (6) test_do_all_referenced_ids_exist: Check if any sub-key (ex: entities/events etc.)
             have referenced keys
         (7) test_passages_offsets: Check if text matches offsets in passages
         (8) test_entities_offsets: Check if text matches offsets in entities
         (9) test_events_offsets: Check if text matches offsets in events
        (10) test_coref_ids: Check if text matches offsets in coreferences

        """  # noqa
        self.setUp()

        # check the schemas implied by _SUPPORTED_TASKS
        if self.SCHEMA is None:
            schemas_to_check = self._MAPPED_SCHEMAS
        # check the schema forced in unit test args
        else:
            schemas_to_check = [self.SCHEMA]

        for schema in schemas_to_check:

            dataset_bigbio = self.datasets_bigbio[schema]
            self.test_are_ids_globally_unique(dataset_bigbio)
            self.test_schema(schema)

            mapped_features = _SCHEMA_TO_FEAUTURES[schema]
            self.print_statistics(mapped_features, schema)

            if schema == "KB":
                self.test_do_all_referenced_ids_exist(dataset_bigbio)
                self.test_passages_offsets(dataset_bigbio)
                self.test_entities_offsets(dataset_bigbio)
                self.test_events_offsets(dataset_bigbio)
                self.test_coref_ids(dataset_bigbio)

    def setUp(self) -> None:
        """Load original and big-bio schema views"""

        logger.info(f"self.PATH: {self.PATH}")
        logger.info(f"self.SUBSET_ID: {self.SUBSET_ID}")
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
            raise ValueError(
                f"Found invalid supported tasks {invalid_tasks}. Must be one of {_VALID_TASKS}"
            )

        self._MAPPED_SCHEMAS = set(
            [_TASK_TO_SCHEMA[task] for task in self._SUPPORTED_TASKS]
        )
        logger.info(f"_SUPPORTED_TASKS implies _MAPPED_SCHEMAS={self._MAPPED_SCHEMAS}")

        config_name = f"{self.SUBSET_ID}_source"
        logger.info(f"Checking load_dataset with config name {config_name}")
        self.dataset_source = datasets.load_dataset(
            self.PATH,
            name=config_name,
            data_dir=self.DATA_DIR,
            use_auth_token=self.USE_AUTH_TOKEN,
        )

        self.datasets_bigbio = {}
        for schema in self._MAPPED_SCHEMAS:
            config_name = f"{self.SUBSET_ID}_bigbio_{schema.lower()}"
            logger.info(f"Checking load_dataset with config name {config_name}")
            self.datasets_bigbio[schema] = datasets.load_dataset(
                self.PATH,
                name=config_name,
                data_dir=self.DATA_DIR,
                use_auth_token=self.USE_AUTH_TOKEN,
            )

    def print_statistics(self, features: Features, schema: str):
        """
        Gets sample statistics, for each split and sample of the number of
        features in the schema present; only works for the big-bio schema.

        :param schema_type: Type of schema to reference features from
        """  # noqa
        logger.info("Gathering schema statistics")
        for split_name, split in self.datasets_bigbio[schema].items():
            print(split_name)
            print("=" * 10)

            counter = defaultdict(int)
            for example in split:
                for feature_name, feature in features.items():
                    if example.get(feature_name, None) is not None:
                        if isinstance(feature, datasets.Value):
                            if example[feature_name]:
                                counter[feature_name] += 1
                        else:
                            counter[feature_name] += len(example[feature_name])

            for k, v in counter.items():
                print(f"{k}: {v}")
            print()

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

    def test_are_ids_globally_unique(self, dataset_bigbio: DatasetDict):
        """
        Tests each example in a split has a unique ID.
        """
        logger.info("Checking global ID uniqueness")
        for split in dataset_bigbio.values():
            ids_seen = set()
            for example in split:
                self._assert_ids_globally_unique(example, ids_seen=ids_seen)
        logger.info("Found {} unique IDs".format(len(ids_seen)))

    def _get_referenced_ids(self, example):
        referenced_ids = []

        if example.get("events", None) is not None:
            for event in example["events"]:
                for argument in event["arguments"]:
                    referenced_ids.append((argument["ref_id"], "event"))

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

    def test_do_all_referenced_ids_exist(self, dataset_bigbio: DatasetDict):
        """
        Checks if referenced IDs are correctly labeled.
        """
        logger.info("Checking if referenced IDs are properly mapped")
        for split in dataset_bigbio.values():
            for example in split:
                referenced_ids = set()
                existing_ids = set()

                referenced_ids.update(self._get_referenced_ids(example))
                existing_ids.update(self._get_existing_referable_ids(example))

                for ref_id, ref_type in referenced_ids:
                    if not (ref_id, ref_type) in existing_ids:
                        logger.warning(
                            f"Referenced element {(ref_id, ref_type)} could not be found in existing ids {existing_ids}. Please make sure that this is not because of a bug in your data loader."
                        )

    def test_passages_offsets(self, dataset_bigbio: DatasetDict):
        """
        Verify that the passages offsets are correct,
        i.e.: passage text == text extracted via the passage offsets
        """  # noqa
        logger.info("KB ONLY: Checking passage offsets")
        for split in dataset_bigbio:

            if "passages" in dataset_bigbio[split].features:

                for example in dataset_bigbio[split]:

                    example_text = _get_example_text(example)

                    for passage in example["passages"]:

                        example_id = example["id"]

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
                            msg = f"Split:{split} - Example:{example_id} - text:`{example_text[start:end]}` != text_by_offset:`{text[idx]}`"
                            self.assertEqual(example_text[start:end], text[idx], msg)

    def _check_offsets(
        self,
        example_id: int,
        split: str,
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
            logger.warning(
                f"Split:{split} - Example:{example_id} - Number of texts {len(texts)} != number of offsets {len(offsets)}. Please make sure that this error already exists in the original data and was not introduced in the data loader."
            )

        self._test_is_list(
            msg=f"Split:{split} - Example:{example_id} - Text fields paired with offsets must be in the form [`text`, ...]",
            field=texts,
        )

        with self.subTest(
            f"Split:{split} - Example:{example_id} - All offsets must be in the form [(lo1, hi1), ...]",
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

    def test_entities_offsets(self, dataset_bigbio: DatasetDict):
        """
        Verify that the entities offsets are correct,
        i.e.: entity text == text extracted via the entity offsets
        """  # noqa
        logger.info("KB ONLY: Checking entity offsets")
        errors = []

        for split in dataset_bigbio:

            if "entities" in dataset_bigbio[split].features:

                for example in dataset_bigbio[split]:

                    example_id = example["id"]
                    example_text = _get_example_text(example)

                    for entity in example["entities"]:

                        for msg in self._check_offsets(
                            example_id=example_id,
                            split=split,
                            example_text=example_text,
                            offsets=entity["offsets"],
                            texts=entity["text"],
                        ):

                            entity_id = entity["id"]
                            errors.append(
                                f"Example:{example_id} - entity:{entity_id} " + msg
                            )

        if len(errors) > 0:
            logger.warning(msg="\n".join(errors) + OFFSET_ERROR_MSG)

    # UNTESTED: no dataset example
    def test_events_offsets(self, dataset_bigbio: DatasetDict):
        """
        Verify that the events' trigger offsets are correct,
        i.e.: trigger text == text extracted via the trigger offsets
        """
        logger.info("KB ONLY: Checking event offsets")
        errors = []

        for split in dataset_bigbio:

            if "events" in dataset_bigbio[split].features:

                for example in dataset_bigbio[split]:

                    example_id = example["id"]
                    example_text = _get_example_text(example)

                    for event in example["events"]:

                        for msg in self._check_offsets(
                            example_id=example_id,
                            split=split,
                            example_text=example_text,
                            offsets=event["trigger"]["offsets"],
                            texts=event["trigger"]["text"],
                        ):

                            event_id = event["id"]
                            errors.append(
                                f"Example:{example_id} - event:{event_id} " + msg
                            )

        if len(errors) > 0:
            logger.warning(msg="\n".join(errors) + OFFSET_ERROR_MSG)

    def test_coref_ids(self, dataset_bigbio: DatasetDict):
        """
        Verify that coreferences ids are entities

        from `examples/test_n2c2_2011_coref.py`
        """  # noqa
        logger.info("KB ONLY: Checking coref offsets")
        for split in dataset_bigbio:

            if "coreferences" in dataset_bigbio[split].features:

                for example in dataset_bigbio[split]:
                    example_id = example["id"]
                    entity_lookup = {ent["id"]: ent for ent in example["entities"]}

                    # check all coref entity ids are in entity lookup
                    for coref in example["coreferences"]:
                        for entity_id in coref["entity_ids"]:
                            assert (
                                entity_id in entity_lookup
                            ), f"Split:{split} - Example:{example_id} - Entity:{entity_id} not found!"

    def test_schema(self, schema: str):
        """Search supported tasks within a dataset and verify big-bio schema"""

        if schema == "KB":

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

            sub_keys = []
            if "NER" in self._SUPPORTED_TASKS:
                sub_keys += ["entities"]
            elif "RE" in self._SUPPORTED_TASKS:
                sub_keys += ["entities", "relations"]
            elif "NED" in self._SUPPORTED_TASKS:
                sub_keys = ["entities"]
            elif "COREF" in self._SUPPORTED_TASKS:
                sub_keys = ["entities", "coreferences"]
            elif "EE" in self._SUPPORTED_TASKS:
                sub_keys = ["events"]
            else:
                raise ValueError(f"Task {task} not recognized")

            logger.info(f"needed_keys: {needed_keys}")
            logger.info(f"sub_keys: {sub_keys}")

            for split in self.datasets_bigbio[schema].keys():
                example = self.datasets_bigbio[schema][split][0]

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

        elif schema == "QA":
            logger.info("Question-Answering Schema")
            self._check_keys(_SCHEMA_TO_FEAUTURES[schema], schema)

        elif schema == "TE":
            logger.info("Textual Entailment Schema")
            self._check_keys(_SCHEMA_TO_FEAUTURES[schema], schema)

        elif schema == "T2T":
            logger.info("Text to Text Schema")
            self._check_keys(_SCHEMA_TO_FEAUTURES[schema], schema)

        elif schema == "TEXT":
            logger.info("Text Schema")
            self._check_keys(_SCHEMA_TO_FEAUTURES[schema], schema)

        elif schema == "PAIRS":
            logger.info("Text Pair Schema")
            self._check_keys(_SCHEMA_TO_FEAUTURES[schema], schema)

        else:
            raise ValueError(
                f"{schema} not recognized. must be one of {set(_TASK_TO_SCHEMA.values())}"
            )

    @staticmethod
    def _check_subkey(inp, attrs):
        """Checks if subkeys (esp. in KB) have necessary criteria"""
        return all([k in inp for k in attrs.keys()])

    def _check_keys(self, features: Features, schema_name: str):
        """Check if necessary keys are present in a given schema"""
        dataset_bigbio = self.datasets_bigbio[schema_name]
        for split in dataset_bigbio.keys():
            example = dataset_bigbio[split][0]

            # Check for mandatory keys
            mandatory_keys = all([key in example for key in features.keys()])
            self.assertTrue(
                mandatory_keys,
                "/".join(features.keys())
                + " keys missing from bigbio view for schema_type = "
                + schema_name,
            )

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Unit tests for BigBio datasets. Args are passed to `datasets.load_dataset`"
    )

    parser.add_argument(
        "path", type=str, help="path to dataloader script (e.g. examples/n2c2_2011.py)"
    )
    parser.add_argument(
        "--schema",
        type=str,
        default=None,
        required=False,
        choices=list(_VALID_SCHEMAS),
        help="by default, bigbio schemas will be discovered from _SUPPORTED_TASKS. use this to explicitly test only one schema.",
    )
    parser.add_argument(
        "--subset_id",
        default=None,
        required=False,
        help="by default, subset_id will be generated from path (e.g. if path=examples/n2c2_2011.py then subset_id=n2c2_2011). the config name is then constructed as config_name=<subset_id>_bigbio_<schema>. use this to explicitly set the subset_id for the config name you want to test (e.g. bioasq9b).",
    ),
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--use_auth_token", default=None)

    args = parser.parse_args()
    logger.info(f"args: {args}")

    if args.subset_id is None:
        subset_id = args.path.split(".py")[0].split("/")[-1]
    else:
        subset_id = args.subset_id

    TestDataLoader.PATH = args.path
    TestDataLoader.SUBSET_ID = subset_id
    TestDataLoader.SCHEMA = args.schema
    TestDataLoader.DATA_DIR = args.data_dir
    TestDataLoader.USE_AUTH_TOKEN = args.use_auth_token

    unittest.TextTestRunner().run(TestDataLoader())
