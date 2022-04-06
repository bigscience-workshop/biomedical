# Guide to Implementing a dataset

## Pre-Requisites

Please make a github account prior to implementing a dataset; you can follow instructions to install git [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

You will also need at least Python 3.6+. If you are installing python, we recommend downloading [anaconda](https://docs.anaconda.com/anaconda/install/index.html) to curate a python environment with necessary packages. **We strongly recommend Python 3.8+ for stability**.

**Optional** Setup your GitHub account with SSH ([instructions here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh).)

### 1. **Setup a local version of the bigscience-biomed repo**
Fork the bigscience-biomed dataset [repository](https://github.com/bigscience-workshop/biomedical) to your local github account. To do this, click the link to the repository and click "fork" in the upper-right corner. You should get an option to fork to your account, provided you are signed into Github.

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

The following instructions will create an Anaconda `bigscience-biomedical` environment.

- Install [anaconda](https://docs.anaconda.com/anaconda/install/) for your appropriate operating system.
- Run the following command while in the `biomedical` folder (you can pick your python version):

```
conda env create -f conda.yml  # Creates a conda env
conda activate bigscience-biomedical  # Activate your conda environment
```

You can deactivate your environment at any time by either exiting your terminal or using `conda deactivate`.

#### 2b) Create a venv environment

Python 3.3+ has venv automatically installed; official information is found [here](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).

```
python3 -m venv <your_env_name_here>
source <your_env_name_here>/bin/activate  # activate environment
pip install -r requirements.txt # Install this while in the datasets folder
```
Make sure your `pip` package points to your environment's source.

### 3. Implement your dataset

Make a new directory within the `biomedical/biodatasets` directory:

    mkdir biodatasets/<dataset_name>

Please use lowercase letters and underscores when choosing a `<dataset_name>`. 
To implement your dataset, there are three key methods that are important:

  * `_info`: Specifies the schema of the expected dataloader
  * `_split_generators`: Downloads and extracts data for each split (e.g. train/val/test) or associate local data with each split.
  * `_generate_examples`: Create examples from data that conform to each schema defined in `_info`.

To start, copy [templates/template.py](templates/template.py) to your `biomedical/biodatasets/<dataset_name>` directory with the name `<dataset_name>.py`. Within this file, fill out all the TODOs.

    cp templates/template.py biodatasets/<dataset_name>/<dataset_name>.py

For the `_info_` function, you will need to define `features` for your
`DatasetInfo` object. For the `bigbio` config, choose the right schema from our list of examples. You can find a description of these in the [Task Schemas Document](task_schemas.md). You can find the actual schemas in the [schemas directory](utils/schemas/).

You will use this schema in the `_generate_examples` return value.

Populate the information in the dataset according to this schema; some fields may be empty.

To enable quality control, please add the following line in your file before the class definition:
```python
from utils.constants import Tasks
_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.RELATION_EXTRACTION]
```

If your dataset is in a standard format, please use a recommended parser if available:
- BioC: Use the excellent [bioc](https://github.com/bionlplab/bioc) package for parsing. Example usage can be found in [examples/bc5cdr.py](examples/bc5cdr.py)
- BRAT: Use [our custom brat parser](utils/parsing.py). Example usage can be found in [examples/mlee.py](examples/mlee.py).

If the recommended parser does not work for you dataset, please alert us in Discord, Slack or the github issue.


##### Example scripts:
To help you implement a dataset, we offer [example scripts](examples/).

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

data = load_dataset("biodatasets/<dataset_name>/<dataset_name>.py", name="<dataset_name>_bigbio_<schema>")
```

Run these commands from the top level of the `biomedical` repo (i.e. the same directory that contains the `requirements.txt` file).

Once this is done, please also check if your dataloader satisfies our unit tests as follows by using this command in the terminal:

```bash
python -m tests.test_bigbio biodatasets/<dataset_name>/<dataset_name>.py [--data_dir /path/to/local/data]
```

Your particular dataset may require use of some of the other command line args in the test script.
To view full usage instructions you can use the `--help` command,

```bash
python -m tests.test_bigbio --help
```

### 5. Format your code

From the main directory, run the Makefile via the following command:

    make check_file=biodatasets/<dataset_name>/<dataset_name>.py

This runs the black formatter, isort, and lints to ensure that the code is readable and looks nice. Flake8 linting errors may require manual changes.

### 6. Commit your changes

First, commit your changes to the branch to "add" the work:

    git add biodatasets/<dataset_name>/<dataset_name>.py
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
