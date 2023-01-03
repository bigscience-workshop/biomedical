# Guide to Implementing a dataset

All dataset loading scripts will be hosted on the [Official `BigBIO` Hub](https://huggingface.co/bigbio). We use this github repository to accept new submissions, and standardize quality control.

## Pre-Requisites

Please make a github account prior to implementing a dataset; you can follow instructions to install git [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

You will also need at least Python 3.6+. If you are installing python, we recommend downloading [anaconda](https://docs.anaconda.com/anaconda/install/index.html) to curate a python environment with necessary packages. **We strongly recommend Python 3.8+ for stability**.

**Optional** Setup your GitHub account with SSH ([instructions here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh).)

### 1. **Fork the BigBIO repository**
Fork the `BigBIO`[repository](https://github.com/bigscience-workshop/biomedical). To do this, click the link to the repository and click "fork" in the upper-right corner. You should get an option to fork to your account, provided you are signed into Github.

After you fork, clone the repository locally. You can do so as follows:

    git clone git@github.com:<your_github_username>/biomedical.git
    cd biomedical  # enter the directory

Next, you want to set your `upstream` location to enable you to push/pull (add or receive updates). You can do so as follows:

    git remote add upstream git@github.com:bigscience-workshop/biomedical.git

You can optionally check that this was set properly by running the following command:

    git remote -v

The output of this command should look as follows:

    origin  git@github.com:<your_github_username>/biomedical.git(fetch)
    origin  git@github.com:<your_github_username>/biomedical.git (push)
    upstream    git@github.com:bigscience-workshop/biomedical.git (fetch)
    upstream    git@github.com:bigscience-workshop/biomedical.git (push)

If you do NOT have an `origin` for whatever reason, then run:

    git remote add origin git@github.com:<your_github_username>/biomedical.git

The goal of `upstream` is to keep your repository up-to-date to any changes that are made officially to the datasets library. You can do this as follows by running the following commands:

    git fetch upstream
    git pull

Provided you have no *merge conflicts*, this will ensure the library stays up-to-date as you make changes. However, before you make changes, you should make a custom branch to implement your changes.

You can make a new branch as such:

    git checkout -b <dataset_name>

<p style="color:red"> <b> Please do not make changes on the master branch! </b></p>

Always make sure you're on the right branch with the following command:

    git branch

The correct branch will have a asterisk \* in front of it.

### 2. **Create a development environment**
You can make an environment in any way you choose to. We highlight two possible options:

#### 2a) Create a conda environment

The following instructions will create an Anaconda `BigBIO` environment.

- Install [anaconda](https://docs.anaconda.com/anaconda/install/) for your appropriate operating system.
- Run the following command while in the `biomedical` folder (you can pick your python version):

```
conda env create -f conda.yml  # Creates a conda env
conda activate bigbio  # Activate your conda environment
```

You can deactivate your environment at any time by either exiting your terminal or using `conda deactivate`.

#### 2b) Create a venv environment

Python 3.3+ has venv automatically installed; official information is found [here](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).

```
python3 -m venv <your_env_name_here>
source <your_env_name_here>/bin/activate  # activate environment
pip install -r dev-requirements.txt # Install this while in the datasets folder
```
Make sure your `pip` package points to your environment's source.

### 3. Prepare the folder in `biodatasets` for your dataloader

Make a new directory within the `biomedical/bigbio/biodatasets` directory:

    mkdir bigbio/biodatasets/<dataset_name>

**NOTE**: Please use lowercase letters and underscores when choosing a `<dataset_name>`.

Add an `__init__.py` file to this directory:

    touch bigbio/biodatasets/<dataset_name>/__init__.py

Next, copy the contents of `template` into your dataset folder. This contains 2 scripts: `bigbiohub.py` that contains all data structures/classes for your dataloader, and `template.py` which has "TODOs" to fill in for your dataloading script.

    cp templates/*.py bigbio/biodatasets/<dataset_name>/<dataset_name>.py


### 4. Implement your dataset

To implement your dataloader, you will need to follow `template.py` and fill in all necessary TODOs. There are three key methods that are important:

  * `_info`: Specifies the schema of the expected dataloader
  * `_split_generators`: Downloads and extracts data for each split (e.g. train/val/test) or associate local data with each split.
  * `_generate_examples`: Create examples from data that conform to each schema defined in `_info`.


For the `_info_` function, you will need to define `features` for your
`DatasetInfo` object. For the `bigbio` config, choose the right schema from our list of examples. You can find a description of these in the [Task Schemas Document](task_schemas.md). You can find the actual schemas in the [schemas directory](bigbio/utils/schemas/).

You will use this schema in the `_generate_examples` return value.

Populate the information in the dataset according to this schema; some fields may be empty.

To enable quality control, please add the following line in your file before the class definition:
```python
from .bigbiohub import Tasks
_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]
```

If your dataset is in a standard format, please use a recommended parser if available:
- BioC: Use the excellent [bioc](https://github.com/bionlplab/bioc) package for parsing. Example usage can be found in [examples/bc5cdr.py](examples/bc5cdr.py)
- BRAT: Use [our custom brat parser](bigbio/utils/parsing.py). Example usage can be found in [examples/mlee.py](examples/mlee.py).

If the recommended parser does not work for you dataset, please alert us in [Discord](https://discord.com/invite/Cwf3nT3ajP), Slack, or a [github issue](https://github.com/bigscience-workshop/biomedical/issues/new?assignees=&labels=&template=add-dataset.md&title=) (please make it a thread in your official project issue).


##### Example scripts:
To help you implement a dataset, we offer [example scripts](examples/). Checkout which task, and [schema](task_schemas.md) best suit your dataset!

#### Running & Debugging:
You can run your data loader script during development by appending the following
statement to your code ([templates/template.py](templates/template.py) already includes this):

```python
if __name__ == "__main__":
    datasets.load_dataset(__file__)
```

If you want to use an interactive debugger during development, you will have to use
`breakpoint()` instead of setting breakpoints directly in your IDE. Most IDEs will
recognize the `breakpoint()` statement and pause there during debugging. If your prefered
IDE doesn't support this, you can always run the script in your terminal and debug with
`pdb`.


### 4. Check if your dataloader works

Make sure your dataset is implemented correctly by checking in python the following commands:

```python
from datasets import load_dataset

data = load_dataset("bigbio/biodatasets/<dataset_name>/<dataset_name>.py", name="<dataset_name>_bigbio_<schema>")
```

Run these commands from the top level of the `biomedical` repo (i.e. the same directory that contains the `requirements.txt` file).

Once this is done, please also check if your dataloader satisfies our unit tests as follows by using this command in the terminal:

```bash
python -m tests.test_bigbio_hub <dataset_name> [--data_dir /path/to/local/data] --test_local
```

You MUST include the `--test_local` flag to specifically test the script for your PR, otherwise the script will default to downloading a dataloader script from the Hub. Your particular dataset may require use of some of the other command line args in the test script (ex: `--data_dir` for dataloaders that read local files).
<br>
To view full usage instructions you can use the `--help` command:

```bash
python -m tests.test_bigbio --help
```
This will explain the types of arguments you may need to test for. A brief annotation is as such:

- `dataset_name`: Name of the dataset you want to test
- `data_dir`: The location of the data for datasets where `LOCAL_ = True`
- `config_name`: Name of the configuration you want to test. By default, the script will test all configs, but if you can use this to debug a specific split, or if your data is prohibitively large.
- `ishub`: Use this when unit testing scripts that are not yet uploaded to the hub (this is True for most cases)

If you need advanced arguments (i.e. skipping a key from a specific data split), please contact admins. You are welcome to make a PR and ask admin for help if your code does not pass the unit tests. 

### 5. Format your code

From the main directory, run the Makefile via the following command:

    make check_file=bigbio/biodatasets/<dataset_name>/<dataset_name>.py

This runs the black formatter, isort, and lints to ensure that the code is readable and looks nice. Flake8 linting errors may require manual changes.

### 6. Commit your changes

First, commit your changes to the branch to "add" the work:

    git add bigbio/biodatasets/<dataset_name>/<dataset_name>.py
    git commit -m "A message describing your commits"

Then, run the following commands to incorporate any new changes in the master branch of datasets as follows:

    git fetch upstream
    git rebase upstream/master

**Run these commands in your custom branch**.

Push these changes to **your fork** with the following command:

    git push -u origin <dataset_name>

### 7. **Make a pull request**

Make a Pull Request to implement your changes on the main repository [here](https://github.com/bigscience-workshop/biomedical/pulls). To do so, click "New Pull Request". Then, choose your branch from your fork to push into "base:master".

When opening a PR, please link the [issue](https://github.com/bigscience-workshop/biomedical/issues) corresponding to your dataset using [closing keywords](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue) in the PR's description, e.g. `resolves #17`.
