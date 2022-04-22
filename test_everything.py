"""
Attempt to load every config of every dataset loader and record the outcome.

* configurable timeout
* catches local datasets

"""

from glob import glob
import os
import time

import datasets
from datasets import load_dataset
import timeout_decorator


# We should probably define a specific Exception for trying to load
# a local dataset w/o passing the data_dir kwarg so we don't have to
# string match
LOCAL_ERR_STR = [
    'This is a local dataset. Please pass the data_dir kwarg to load_dataset.',
    'This is a local dataset. Please pass the data_dir and name kwarg to load_dataset.',
]

# maximum time in seconds before stopping
MAX_TIME_FOR_LOADING = 60


@timeout_decorator.timeout(MAX_TIME_FOR_LOADING)
def read_bigbio_dataset(loader_script, name):
    ds = load_dataset(loader_script, name=name)
    return ds


finished_configs = []
timedout_configs = []
local_configs = []
error_configs = []


loader_scripts = sorted(glob(os.path.join("biodatasets", "*", "*.py")))
for loader_script in loader_scripts:

    module = datasets.load.dataset_module_factory(loader_script)
    builder_cls = datasets.load.import_main_class(module.module_path)
    config_names = [el.name for el in builder_cls.BUILDER_CONFIGS]

    for config_name in config_names:

        # here the config works
        try:
            ds = read_bigbio_dataset(loader_script, config_name)
            finished_configs.append((loader_script, config_name))

        # here we hit the max time
        except timeout_decorator.TimeoutError:
            print("Loading timeout")
            timedout_configs.append((loader_script, config_name))

        # here we check for local dataset
        except ValueError as oops:
            if len(oops.args) == 1 and oops.args[0] in LOCAL_ERR_STR:
                local_configs.append((loader_script, config_name))
                continue
            else:
                raise

        # here some other error is raised
        except:
            error_configs.append((loader_script, config_name))


# TODO: do something with the breakdown of
# * finished, timedout, local, error
