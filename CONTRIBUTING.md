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

    git checkout -b <name_of_my_dataset_branch>

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
conda env create -f environment.yml  # Creates a conda env
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

Make a new directory within the `biomedical/datasets` folder as such: <br>

    mkdir datasets/<name_of_my_dataset>
    cd datasets/<name_of_my_dataset>

To implement your dataset, there are three key methods that are important:<br>

  * `_info`: Specifies the schema of the expected dataloader
  * `_split_generators`: Downloads and extracts data for each split (e.g. train/val/test) or associate local data with each split.
  * `_generate_examples`: Create examples from data that conform to each schema defined in `_info`.

To start, copy [templates/template.py](templates/template.py) to your in `biomedical/datasets/<name_of_my_dataset>` with the name <name_of_my_dataset>.py. Within this file, fill out all the TODOs.


For the `_info_` function, you will need to define `features` for your
`DatasetInfo` object. For the `big-bio` config, copy the right schema from our list of examples. You can find them as follows:

1. [Named entity recognition (NER)](https://github.com/bigscience-workshop/biomedical/blob/master/schemas/kb.py)
2. [Relation Extraction (RE)](https://github.com/bigscience-workshop/biomedical/blob/master/schemas/kb.py)
3. [Event Extraction](https://github.com/bigscience-workshop/biomedical/blob/master/schemas/kb.py)
4. [Named entity disambiguation/canonicalization/normalization (NED)](https://github.com/bigscience-workshop/biomedical/blob/master/schemas/kb.py)
5. [Co-reference resolution](https://github.com/bigscience-workshop/biomedical/blob/master/schemas/kb.py)
6. [Question-Answering](https://github.com/bigscience-workshop/biomedical/blob/aster/schemas/qa.py)
7. [Entailment](https://github.com/bigscience-workshop/biomedical/blob/master/schemas/entailment.py)
8. [Translation](https://github.com/bigscience-workshop/biomedical/blob/master/schemas/text_to_text.py)
9. [Summarization](https://github.com/bigscience-workshop/biomedical/blob/master/schemas/text_to_text.py)
10. [Paraphrasing](https://github.com/bigscience-workshop/biomedical/blob/master/schemas/text_to_text.py)
11. [Sentence/Phrase/Text classification](https://github.com/bigscience-workshop/biomedical/blob/master/schemas/text.py)
12. [Pair Labels](https://github.com/bigscience-workshop/biomedical/blob/master/schemas/pairs.py)

You will use this schema in the `_generate_examples` return value.

Please read the [Task Schemas](task_schemas.md) to understand how each key should behave.

Populate the information in the dataset according to this schema; some fields may be empty.

To enable quality control, please add the following line in your file before the class definition:
```python
_SUPPORTED_TASKS = ["your_task_name_here"]
```
For ease, please refer to the following keywords, given your NLP task:
1. **NER, RE, Event Extraction, NED/ Coreference resolution**: `"kb"`
2. **Question-Answering**:"" `"qa"`
3. **Entailment**: `"entailment"`
4. **Translation, Summarization, Paraphrasing**: `"text_to_text"`
5. **Sentence/Phrase/Text Classification**: `"text"`
6. **Pair Labels**: `"pairs"`

##### Example scripts:
To help you implement a dataset, we offer a template and example scripts.


### 4. Check if your dataloader works

Make sure your dataset is implemented correctly by checking in python the following commands:

```python
import datasets
from datasets import load_dataset

data = load_dataset('biomeddatasets/<your_dataset_name>', name="bigbio")
```

Run these commands within the `biomedical` repo, not in `biomedical/datasets` as the relative path will not work.

Once this is done, please also check if your dataloader satisfies our unit tests as follows by using this command in the terminal:

```
python -m tests/test_bigbio --path biomeddatasets/<your_dataset_name> [--data_dir /path/to/local/data]
```

### 5. Format your code

Return to the main directory (assuming you are in your dataset-specific folder) via the following commands:

    cd ../../  # return to the datasets main location
    make style
    make quality

This runs the black formatter, isort, and lints to ensure that the code is readable and looks nice. Flake8 linting errors may require manual changes.

### 6. Commit your changes

First, commit your changes to the branch to "add" the work:

    git add datasets/<name_of_my_dataset>/<name_of_my_dataset>.py
    git commit

*Ideally, run* `git commit -m "A message of your commits"`

Then, run the following commands to incorporate any new changes in the master branch of datasets as follows:

    git fetch upstream
    git rebase upstream/master

**Run these commands in your custom branch**.

Push these changes to **your fork** with the following command:

    git push -u origin <name_of_my_dataset_branch>

**[Optional Step]** Add unit-tests and meta data by following instructions [here](https://huggingface.co/docs/datasets/share_dataset.html#adding-tests).

### 7. **Make a pull request**

Make a PR to implement your changes on the main repository [here](https://github.com/huggingface/datasets/pulls). To do so, click "New Pull Request". Then, choose your branch from your fork to push into "base:master".
