"""
Unit-tests to ensure tasks adhere to big-bio schema.
"""
import argparse
from collections import defaultdict
import importlib
import logging
from pathlib import Path
import re
import sys
from typing import Iterable, Iterator, List, Optional, Union, Dict
import unittest

import datasets
from datasets import DatasetDict, Features
from bigbio.utils.constants import Tasks
from bigbio.utils.schemas import (
    entailment_features,
    kb_features,
    pairs_features,
    qa_features,
    text2text_features,
    text_features,
)


logger = logging.getLogger(__name__)


_TASK_TO_SCHEMA = {
    Tasks.NAMED_ENTITY_RECOGNITION: "KB",
    Tasks.NAMED_ENTITY_DISAMBIGUATION: "KB",
    Tasks.EVENT_EXTRACTION: "KB",
    Tasks.RELATION_EXTRACTION: "KB",
    Tasks.COREFERENCE_RESOLUTION: "KB",
    Tasks.QUESTION_ANSWERING: "QA",
    Tasks.TEXTUAL_ENTAILMENT: "TE",
    Tasks.SEMANTIC_SIMILARITY: "PAIRS",
    Tasks.PARAPHRASING: "T2T",
    Tasks.TRANSLATION: "T2T",
    Tasks.SUMMARIZATION: "T2T",
    Tasks.TEXT_CLASSIFICATION: "TEXT",
}

_VALID_TASKS = set(_TASK_TO_SCHEMA.keys())
_VALID_SCHEMAS = set(_TASK_TO_SCHEMA.values())

_SCHEMA_TO_FEATURES = {
    "KB": kb_features,
    "QA": qa_features,
    "TE": entailment_features,
    "T2T": text2text_features,
    "TEXT": text_features,
    "PAIRS": pairs_features,
}

_TASK_TO_FEATURES = {
    Tasks.NAMED_ENTITY_RECOGNITION: {"entities"},
    Tasks.RELATION_EXTRACTION: {"relations", "entities"},
    Tasks.NAMED_ENTITY_DISAMBIGUATION: {"entities", "normalized"},
    Tasks.COREFERENCE_RESOLUTION: {"entities", "coreferences"},
    Tasks.EVENT_EXTRACTION: {"events"},
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

_CONNECTORS = re.compile(r"\+|\,|\||\;")


class TestDataLoader(unittest.TestCase):
    """
    Test a single config from a dataloader script.
    """

    PATH: str
    NAME: str
    DATA_DIR: Optional[str]
    BYPASS_SPLITS: List[str]
    BYPASS_KEYS: List[str]
    BYPASS_SPLIT_KEY_PAIRS: List[str]

    def runTest(self):

        logger.info(f"self.PATH: {self.PATH}")
        logger.info(f"self.NAME: {self.NAME}")
        logger.info(f"self.DATA_DIR: {self.DATA_DIR}")

        self._warn_bypass()

        logger.info("importing module .... ")
        module_name = self.PATH
        if module_name.endswith(".py"):
            module_name = module_name[:-3]
        module_name = module_name.replace("/", ".")
        module = importlib.import_module(module_name)
        logger.info(f"imported module {module}")


        logger.info("Checking for _SUPPORTED_TASKS ...")
        self._SUPPORTED_TASKS = module._SUPPORTED_TASKS
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


        logger.info(f"Checking load_dataset with config name {config_name}")
        self.dataset = datasets.load_dataset(
            self.PATH,
            name=self.NAME,
            data_dir=self.DATA_DIR,
        )

        if "bigbio" in self.NAME:
            schema = self.NAME.split("_")[-1].upper()
        else:
            schema = "source"

        logger.info(f"schema = {schema}")
        if schema == "source":
            return

        with self.subTest("IDs globally unique"):
            self.test_are_ids_globally_unique(self.dataset)
        with self.subTest("Check schema validity"):
            self.test_schema(schema)

        if schema == "KB":
            with self.subTest("Check referenced ids"):
                self.test_do_all_referenced_ids_exist(self.dataset)
            with self.subTest("Check passage offsets"):
                self.test_passages_offsets(self.dataset)
            with self.subTest("Check entity offsets"):
                self.test_entities_offsets(self.dataset)
                self.test_entities_multilabel_db_id(self.dataset)
            with self.subTest("Check events offsets"):
                self.test_events_offsets(self.dataset)
            with self.subTest("Check coref offsets"):
                self.test_coref_ids(self.dataset)
            with self.subTest("Check multi-label `type`"):
                self.test_multilabel_type(self.dataset)

        elif schema == "QA":
            with self.subTest("Check multiple choice"):
                self.test_multiple_choice(self.dataset)


    def get_feature_statistics(self, features: Features) -> Dict:
        """
        Gets sample statistics, for each split and sample of the number of
        features in the schema present; only works for the big-bio schema.
        """  # noqa
        logger.info("Gathering dataset statistics")
        all_counters = {}
        for split_name, split in self.dataset.items():

            counter = defaultdict(int)
            for example in split:
                for feature_name, feature in features.items():
                    if example.get(feature_name, None) is not None:
                        if isinstance(feature, datasets.Value):
                            if example[feature_name]:
                                counter[feature_name] += 1
                        else:
                            counter[feature_name] += len(example[feature_name])

                            # TODO do proper recursion here
                            if feature_name == "entities":
                                for entity in example["entities"]:
                                    counter["normalized"] += len(entity["normalized"])

            all_counters[split_name] = counter

        return all_counters

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
        for split_name, split in dataset_bigbio.items():

            # skip entire split
            if split_name in self.BYPASS_SPLITS:
                logger.info(f"\tSkipping unique ID check on {split_name}")
                continue  

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

            # skip entire split
            if split in self.BYPASS_SPLITS:
                logger.info(f"\tSkipping referenced ids on {split}")
                continue  

            for example in split:
                referenced_ids = set()
                existing_ids = set()


                referenced_ids.update(self._get_referenced_ids(example))
                existing_ids.update(self._get_existing_referable_ids(example))

                for ref_id, ref_type in referenced_ids:
                    if ref_type == "event":
                        if not (
                            (ref_id, "entity") in existing_ids
                            or (ref_id, "event") in existing_ids
                        ):
                            logger.warning(
                                f"Referenced element ({ref_id}, entity/event) could not be "
                                f"found in existing ids {existing_ids}. Please make sure that "
                                f"this is not because of a bug in your data loader."
                            )
                    else:
                        if not (ref_id, ref_type) in existing_ids:
                            logger.warning(
                                f"Referenced element {(ref_id, ref_type)} could not be "
                                f"found in existing ids {existing_ids}. Please make sure that "
                                f"this is not because of a bug in your data loader."
                            )

    def test_passages_offsets(self, dataset_bigbio: DatasetDict):
        """
        Verify that the passages offsets are correct,
        i.e.: passage text == text extracted via the passage offsets
        """  # noqa
        logger.info("KB ONLY: Checking passage offsets")
        for split in dataset_bigbio:

            # skip entire split
            if split in self.BYPASS_SPLITS:
                logger.info(f"\tSkipping passage offsets on {split}")
                continue  

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
                            msg = (
                                f"Split:{split} - Example:{example_id} - "
                                f"text:`{example_text[start:end]}` != text_by_offset:`{text[idx]}`"
                            )
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
                f"Split:{split} - Example:{example_id} - "
                f"Number of texts {len(texts)} != number of offsets {len(offsets)}. "
                f"Please make sure that this error already exists in the original "
                f"data and was not introduced in the data loader."
            )

        self._test_is_list(
            msg=(
                f"Split:{split} - Example:{example_id} - "
                f"Text fields paired with offsets must be in the form [`text`, ...]"
            ),
            field=texts,
        )

        with self.subTest(
            (
                f"Split:{split} - Example:{example_id} - "
                f"All offsets must be in the form [(lo1, hi1), ...]"
            ),
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

            # skip entire split
            if split in self.BYPASS_SPLITS:
                logger.info(f"\tSkipping entities offsets on {split}")
                continue  

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


    def test_events_offsets(self, dataset_bigbio: DatasetDict):
        """
        Verify that the events' trigger offsets are correct,
        i.e.: trigger text == text extracted via the trigger offsets
        """
        logger.info("KB ONLY: Checking event offsets")
        errors = []

        for split in dataset_bigbio:

            # skip entire split
            if split in self.BYPASS_SPLITS:
                logger.info(f"\tSkipping events offsets on {split}")
                continue  

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

            # skip entire split
            if split in self.BYPASS_SPLITS:
                logger.info(f"\tSkipping coref ids on {split}")
                continue  

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

    def test_multiple_choice(self, dataset_bigbio: DatasetDict):
        """
        Verify that each answer in a multiple choice Q/A task is in choices.
        """
        logger.info("QA ONLY: Checking multiple choice")
        for split in dataset_bigbio:

            # skip entire split
            if split in self.BYPASS_SPLITS:
                logger.info(f"\tSkipping multiple-choice on {split}")
                continue  

            for example in dataset_bigbio[split]:

                if ("choices" in self.BYPASS_KEYS) or ( (split, "choices") in self.BYPASS_SPLIT_KEY_PAIRS):

                    logger.warning("Skipping multiple choice for key=choices, split='{split}'")
                    continue
                else:

                    if len(example["choices"]) > 0:
                        # can change "==" to "in" if we include ranking later
                        assert (
                            example["type"] == "multiple_choice"
                        ), f"`choices` is populated, but type is not 'multiple_choice' {example}"

                    if example["type"] == "multiple_choice":
                        assert (
                            len(example["choices"]) > 0
                        ), f"type is 'multiple_choice' but no values in 'choices' {example}"


                        if ("answer" in self.BYPASS_KEYS) or ( (split, "answer") in self.BYPASS_SPLIT_KEY_PAIRS):

                            logger.warning("Skipping multiple choice for key=answer, split='{split}'")
                            continue
                            
                        else:
                            for answer in example["answer"]:
                                assert (
                                    answer in example["choices"]
                                ), f"answer is not present in 'choices' {example}"


    def test_entities_multilabel_db_id(self, dataset_bigbio: DatasetDict):
        """
        Check if `db_id` of `normalized` field in entities have multiple values joined with common connectors.
        """
        logger.info("KB ONLY: multi-label `db_id`")

        warning_raised = False

        # yeah it looks bad: the idea is to avoid to go through the entire dataset
        # one warning is enough to prompt a cleaning pass
        for split in dataset_bigbio:

            # skip entire split
            if split in self.BYPASS_SPLITS:
                logger.info(f"\tSkipping entities multilabel db on {split}")
                continue


            if warning_raised:
                break

            if "entities" not in dataset_bigbio[split].features:
                continue

            if "entities" in self.BYPASS_KEYS:
                logger.warning(f"Skipping multilabel type for key = entities")
                continue

            if (split, "entities") in self.BYPASS_SPLIT_KEY_PAIRS:
                logger.warning(f"Skipping multilabel type for key=entities, split='{split}'")
                continue

            for example in dataset_bigbio[split]:

                if warning_raised:
                    break

                example_id = example["id"]

                for entity in example["entities"]:

                    if warning_raised:
                        break

                    normalized = entity.get("normalized", [])
                    entity_id = entity["id"]

                    for norm in normalized:

                        db_id = norm["db_id"]
                        match = re.search(_CONNECTORS, db_id)

                        if match is not None:

                            connector = match.group(0)

                            logger.warning(
                                f"Split:{split} - Example:{example_id} - ",
                                f"Entity:{entity_id} contains a normalization with a connector `{connector}`",
                                "Please make sure you are that you are expanding the normalization list for each `db_id`",
                            )

                            warning_raised = True

    def test_multilabel_type(self, dataset_bigbio: DatasetDict):
        """
        Check if features with `type` field contain multilabel values
        and raise a warning ONLY ONCE for feature type (e.g. passages)
        """

        logger.info("KB ONLY: multi-label `type` fields")

        features_with_type = ["passages", "entities", "relations", "events"]

        warning_raised = {f: False for f in features_with_type}

        for split in dataset_bigbio:

            # skip entire split
            if split in self.BYPASS_SPLITS:
                logger.info(f"\tSkipping multilabel type on {split}")
                continue  

            for feature_name in features_with_type:

                if feature_name in self.BYPASS_KEYS:
                    logger.warning(f"Skipping multilabel type for key = '{feature_name}'")
                    continue

                if (split, feature_name) in self.BYPASS_SPLIT_KEY_PAIRS:
                    logger.warning(f"Skipping multilabel type for splitkey = '{(split, feature_name)}'")
                    continue

                if (
                    feature_name not in dataset_bigbio[split].features
                    or warning_raised[feature_name]
                ):
                    continue

                for example in dataset_bigbio[split]:

                    if warning_raised[feature_name]:
                        break

                    example_id = example["id"]
                    features = example[feature_name]

                    for feature in features:

                        feature_type = feature["type"]
                        match = re.search(_CONNECTORS, feature_type)

                        if match is not None:

                            connector = match.group(0)

                            logger.warning(
                                f"Split{split} - Example{example_id} - ",
                                f"Feature:{feature_name} contains a connector `{connector}` \n",
                                "Having multiple types it is currently not supported.",
                                "Please split this featuere into multiple ones with different `type`",
                            )

                            warning_raised[feature_name] = True

                            break


    def test_schema(self, schema: str):
        """Search supported tasks within a dataset and verify big-bio schema"""

        non_empty_features = set()
        if schema == "KB":
            features = kb_features
            for task in self._SUPPORTED_TASKS:
                if task in _TASK_TO_FEATURES:
                    non_empty_features.update(_TASK_TO_FEATURES[task])
        else:
            features = _SCHEMA_TO_FEATURES[schema]

        split_to_feature_counts = self.get_feature_statistics(features=features)

        # Skip ALL bypassed keys
        for key in self.BYPASS_KEYS:
            if key in non_empty_features:
                logger.warning(f"Skipping schema for key = '{key}'")
                non_empty_features.remove(key)

        for split_name, split in self.dataset.items():
            print(split_name)
            print("=" * 10)
            for k, v in split_to_feature_counts[split_name].items():
                print(f"{k}: {v}")
            print()

        for split_name, split in self.dataset.items():

            # Skip entire data split
            if split_name in self.BYPASS_SPLITS:
                logger.info(f"Skipping schema on {split_name}")
                continue  

            logger.info("Testing schema for: " + str(split_name))
            self.assertEqual(split.info.features, features)

            for non_empty_feature in non_empty_features:

                # Args for BYPASS_SPLIT_KEY_PAIRS
                split_key = [split_name, non_empty_feature]
                if split_key in self.BYPASS_SPLIT_KEY_PAIRS:
                    logger.info(f"Skipping schema for '{split_key}'")
                    continue

                if (split_to_feature_counts[split_name][non_empty_feature] == 0):
                    raise AssertionError(
                        f"Required key '{non_empty_feature}' does not have any instances"
                    )

            for feature, count in split_to_feature_counts[split_name].items():
                if (
                    count > 0
                    and feature not in non_empty_features
                    and feature in set().union(*_TASK_TO_FEATURES.values())
                ):
                    logger.warning(
                        f"Found instances of '{feature}' but there seems to be no task "
                        f"in 'SUPPORTED_TASKS' for them. Is 'SUPPORTED_TASKS' correct?"
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

    def _warn_bypass(self):
        """ Warn if keys, data splits, or schemas are skipped """

        if len(self.BYPASS_SPLITS) > 0:
            logger.warning(f"Splits ignored = '{self.BYPASS_SPLITS}'")

        if len(self.BYPASS_KEYS) > 0:
            logger.warning(f"Keys ignored = '{self.BYPASS_KEYS}'")

        if len(self.BYPASS_SPLIT_KEY_PAIRS) > 0:
            logger.warning(
                f"Split and key pairs ignored ='{self.BYPASS_SPLIT_KEY_PAIRS}'"
            )
            self.BYPASS_SPLIT_KEY_PAIRS = [i.split(",") for i in self.BYPASS_SPLIT_KEY_PAIRS]



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Unit tests for BigBio dataloaders.")

    parser.add_argument(
        "dataloader_path",
        type=str,
        help="path to dataloader script (e.g. biodatasets/n2c2_2011/n2c2_2011.py)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="path to local data",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        help="use to run on a single config name (default is to run on all config names)",
    )

    parser.add_argument(
        "--bypass_splits",
        default=[],
        required=False,
        nargs="*",
        help="Skip a data split (e.g. 'train', 'dev') from testing. List all splits as space separated (ex: --bypass_splits train dev)",
    )

    parser.add_argument(
        "--bypass_keys",
        default=[],
        required=False,
        nargs="*",
        help="Skip a required key (e.g. 'entities' for NER) from testing. List all keys as space separated (ex: --bypass_keys entities events)",
    )

    parser.add_argument(
        "--bypass_split_key_pairs",
        default=[],
        required=False,
        nargs="*",
        help="Skip a key in a data split (e.g. skip 'entities' in 'test'). List all key-pairs comma separated. (ex: --bypass_split_key_pairs test,entities train, events)",
    )


    args = parser.parse_args()
    logger.info(f"args: {args}")

    module = datasets.load.dataset_module_factory(args.dataloader_path)
    builder_cls = datasets.load.import_main_class(module.module_path)
    all_config_names = [el.name for el in builder_cls.BUILDER_CONFIGS]
    logger.info(f"all_config_names: {all_config_names}")

    if args.config_name is not None:
        run_config_names = [args.config_name]
    else:
        run_config_names = all_config_names

    for config_name in run_config_names:
        TestDataLoader.PATH = args.dataloader_path
        TestDataLoader.NAME = config_name
        TestDataLoader.DATA_DIR = args.data_dir
        TestDataLoader.BYPASS_SPLITS = args.bypass_splits
        TestDataLoader.BYPASS_KEYS = args.bypass_keys
        TestDataLoader.BYPASS_SPLIT_KEY_PAIRS = args.bypass_split_key_pairs
        unittest.TextTestRunner().run(TestDataLoader())
