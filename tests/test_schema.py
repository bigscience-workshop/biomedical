"""
Unit-tests to ensure tasks adhere to QA for big-bio schema
"""
import os
import sys
import glob

import datasets
from datasets import load_dataset
from ..schemas import (
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
from typing import Iterator, Optional, Union, Iterable

sys.path.append(str(Path(__file__).parent.parent))

import logging as log

log.basicConfig(level=log.INFO)


_TASK_MAPPING = {
    "kb": "kb",
    "ner": "kb",
    "re": "kb",
    "events": "kb",
    "coref": "kb",
    "qa": "qa",
    "question-answering": "qa",
    "entailment": "entailment",
    "paraphrasing": "text_to_text",
    "summarization": "text_to_text",
    "translation": "text_to_text",
    "text_to_text": "text_to_text",
    "classification": "text",
    "text": "text",
    "pairs": "pairs",
}


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
        self.print_statistics()
        self.test_are_ids_globally_unique()
        self.test_do_all_referenced_ids_exist()
        self.test_passages_offsets()
        self.test_entities_offsets()
        self.test_events_offsets()
        self.test_coref_ids()
        self.test_schema()

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

        self._SUPPORTED_TASKS = importlib.import_module(
            "biomeddatasets." + self.NAME + "." + self.NAME
        )._SUPPORTED_TASKS

    def print_statistics(self):
        """
        Gets sample statistics, for each split and sample of the number of features in the schema present; only works for the big-bio schema.
        """
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

    def test_are_ids_globally_unique(self):
        """
        Tests each example in a split has a unique ID
        """
        for split in self.dataset_bigbio.values():
            ids_seen = set()
            for example in split:
                self._assert_ids_globally_unique(example, ids_seen=ids_seen)

    def test_do_all_referenced_ids_exist(self):
        """ """
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
                        self.assertTrue(
                            (ref_id, "entity") in existing_ids
                            or (ref_id, "event") in existing_ids
                        )
                    else:
                        self.assertIn((ref_id, ref_type), existing_ids)

    def _assert_ids_globally_unique(
        self,
        collection: Iterable,
        ids_seen: set,
        ignore_assertion: bool = False,
    ):
        """
        Checks if all IDs are globally unique across a feature list.

        :param collection: An iterable of features that contain NLP info
        :param ids_seen: Empty set of numerical ids
        :param ignore_assertion:
        """
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
        """
        Given a KB-schema, check referenced ids

        :param example: A feature instance in a schema
        """
        referenced_ids = []

        for event in example.get("events", []):
            args = event.get("arguments", None)
            if args is not None:
                for argument in args:
                    referenced_ids.append((argument["ref_id"], "event_arg"))

        for coreference in example.get("coreferences", []):
            args = coreference.get("entity_ids", None)
            if args is not None:
                for entity_id in args:
                    referenced_ids.append((entity_id, "entity"))

        for relation in example.get("relations", []):
            referenced_ids.append((relation["arg1_id"], "entity"))
            referenced_ids.append((relation["arg2_id"], "entity"))

        return referenced_ids

    def _get_existing_referable_ids(self, example):
        """"""
        existing_ids = []

        for entity in example.get("entities", []):
            existing_ids.append((entity["id"], "entity"))

        for event in example.get("events", []):
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
                            self.assertEqual(
                                example_text[start:end], text[idx]
                            )

    def _check_offsets(
        self,
        example_text: str,
        offsets: list[list[int]],
        texts: list[str],
    ) -> Iterator:
        """ """

        with self.subTest(
            "# of texts must be equal to # of offsets",
            texts=texts,
            offsets=offsets,
        ):
            self.assertEqual(len(texts), len(offsets))

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
                                f"Example:{example_id} - entity:{entity_id} "
                                + msg
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
                                f"Example:{example_id} - event:{event_id} "
                                + msg
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
        """
        datafolder = os.path.join("biomeddatasets/", self.NAME)
        datascript = os.path.join(
            "biomeddatasets/", self.NAME, self.NAME + ".py"
        )
        self.assertTrue(
            os.path.exists(datafolder), "Folder not named " + self.NAME
        )

        self.assertTrue(
            os.path.exists(datascript),
            "Script not named " + self.NAME + ".py",
        )

    def test_schema(self):
        """Search supported tasks within a dataset"""

        # Import the _SUPPORTED_TASKS

        for task in self._SUPPORTED_TASKS:

            mapped_task = _TASK_MAPPING.get(task, None)

            self.assertFalse(
                mapped_task is None, task + " is not recognized"
            )

            if mapped_task == "kb":
                log.info("NER/NED/RE/Event Extraction/Coref Task")

                schema = kb_features

                # TODO - change this
                opt_keys = [
                    "entities",
                    "events",
                    "coreferences",
                    "relations",
                ]

                for split in self.dataset_bigbio.keys():
                    example = self.dataset_bigbio[split][0]

                    # Check for mandatory keys
                    mandatory_keys = all(
                        [
                            key in example
                            for key in schema.keys()
                            if key not in opt_keys
                        ]
                    )
                    self.assertTrue(
                        mandatory_keys,
                        "id/document_id/passages missing from bigbio view",
                    )

                    subkeys = []

                    # Ensure tasks that require certain fields are present
                    if task == "ner" or task == "re":
                        subkeys += ["entities", "relations"]

                        for key in subkeys:
                            for attrs in schema[key]:

                                self.assertTrue(
                                    all(
                                        [
                                            k in example[key][0]
                                            for k in attrs.keys()
                                        ]
                                    )
                                )
                    elif task == "coref":
                        subkeys += ["coreferences"]

                        for key in subkeys:
                            for attrs in schema[key]:

                                self.assertTrue(
                                    all(
                                        [
                                            k in example[key][0]
                                            for k in attrs.keys()
                                        ]
                                    )
                                )

                    elif task == "events":
                        subkeys += ["events"]

                        for key in subkeys:
                            for attrs in schema[key]:

                                self.assertTrue(
                                    all(
                                        [
                                            k in example[key][0]
                                            for k in attrs.keys()
                                        ]
                                    )
                                )
                    else:
                        # miscellaneous keys
                        extra_keys = [
                            k for k in example.keys() if k not in subkeys
                        ]
                        for key in extra_keys:
                            if key in schema.keys():
                                self.assertTrue(
                                    all(
                                        [
                                            k in example[key][0]
                                            for k in attrs.keys()
                                        ]
                                    )
                                )

            elif mapped_task == "qa":
                log.info("Question-Answering Task")
                schema = qa_features
                self.check_keys(schema)

            elif mapped_task == "entailment":
                log.info("Entailment Task")
                schema = entailment_features
                self.check_keys(schema)

            elif mapped_task == "text_to_text":
                log.info("Translation/Summarization/Paraphrasing Task")
                schema = text2text_features
                self.check_keys(schema)

            elif mapped_task == "text":
                log.info("Sentence/Phrase/Text Classification Task")
                schema = text_features
                self.check_keys(schema)
                
            elif mapped_task == "pairs":
                log.info("Pair Labels Task")
                schema = pairs_features
                self.check_keys(schema)

            else:
                raise ValueError(
                    task
                    + " not specified; only 'kb', 'qa', 'entailment', 'text_to_text', 'text', 'pairs' allowed"
                )
    def check_keys(self, schema):
        """ Check if necessary keys are present """
        for split in self.dataset_bigbio.keys():
            example = self.dataset_bigbio[split][0]

            # Check for mandatory keys
            mandatory_keys = all(
                [key in example for key in schema.keys()]
            )
            self.assertTrue(
                mandatory_keys,
                "/".join(schema.keys()) + " keys missing from bigbio view",
            )
