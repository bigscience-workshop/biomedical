"""
Attempt to load every config of every dataset loader and record the outcome.

* configurable timeout
* catches local datasets

"""

import json
import os
import time
from collections import defaultdict
from glob import glob

import datasets
import timeout_decorator
from datasets import load_dataset

# We should probably define a specific Exception for trying to load
# a local dataset w/o passing the data_dir kwarg so we don't have to
# string match
LOCAL_ERR_STR = [
    "This is a local dataset. Please pass the data_dir kwarg to load_dataset.",
    "This is a local dataset. Please pass the data_dir and name kwarg to load_dataset.",
]

# maximum time in seconds before stopping
MAX_TIME_FOR_LOADING = 300


@timeout_decorator.timeout(MAX_TIME_FOR_LOADING)
def read_bigbio_dataset(script, name):

    return load_dataset(script, name=name)


def get_source_config_names(config_names: list[str]) -> list[str]:

    config_names = [name for name in config_names if "source" in name]

    source_config_names = config_names if len(config_names) > 0 else []

    return source_config_names


def get_logs(path, names):

    logs = {}

    for name in names:

        logs[name] = {}

        filepath = os.path.join(path, f"{name}.json")

        if os.path.exists(filepath):

            with open(filepath) as infile:
                for line in infile:
                    d = json.loads(line)
                    for k, v in d.items():
                        logs[name][k] = v

    return logs


def add_to_log(item: dict, path: str):

    with open(path, "a") as outfile:

        outfile.write(f"{json.dumps(item, indent=1)}\n")


logs_dir = os.path.join(os.getcwd(), "loading_logs")
os.makedirs(logs_dir, exist_ok=True)

log_names = [
    "timedout",
    "local",
    "error",
    "no_source",
    "no_bigbio",
]

logs = get_logs(path=logs_dir, names=log_names)

# logs_handles : dict = {name: os.path.join(logs_dir, f"{name}.json") for name in logs}

loader_scripts = sorted(glob(os.path.join("biodatasets", "*", "*.py")))

for loader_script in loader_scripts:

    if "__init__.py" in loader_script:
        continue

    if any(loader_script in log_dict for log_name, log_dict in logs.items()):
        continue

    module = datasets.load.dataset_module_factory(loader_script)
    builder_cls = datasets.load.import_main_class(module.module_path)
    config_names = [el.name for el in builder_cls.BUILDER_CONFIGS]

    source_config_names = get_source_config_names(config_names=config_names)

    if len(source_config_names) == 0:
        add_to_log(
            item={loader_script: config_names},
            path=os.path.join(logs_dir, "no_source.json"),
        )
        continue

    if len([c for c in config_names if c not in source_config_names]) == 0:
        add_to_log(
            item={loader_script: config_names},
            path=os.path.join(logs_dir, "no_bigbio.json"),
        )

    tmp_logs: dict = {k: defaultdict(set) for k in ["timedout", "local", "error"]}

    for config_name in source_config_names:

        try:
            ds = read_bigbio_dataset(loader_script, config_name)
            # here we hit the max time
        except timeout_decorator.TimeoutError:
            tmp_logs["timedout"][loader_script].add(config_name)

        # here we check for local dataset
        except ValueError as oops:
            if len(oops.args) == 1 and oops.args[0] in LOCAL_ERR_STR:
                tmp_logs["local"][loader_script].add(config_name)
                continue
            else:
                tmp_logs["error"][loader_script].add((config_name, str(oops)))

        # here some other error is raised
        except Exception as e:
            tmp_logs["error"][loader_script].add((config_name, str(e)))

    for log_name, item in tmp_logs.items():
        if len(item) > 0:
            add_to_log(
                item={k: list(v) for k, v in item.items()},
                path=os.path.join(logs_dir, f"{log_name}.json"),
            )
