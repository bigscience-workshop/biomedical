# Biomedical datasets hackathon

This hackathon aims to implement several biomedical datasets within Huggingface's `datasets` library. This library contains many different datasets and the unique natural language attributes that describe them. Biological data can be quite diverse, so a unified location that joins multiple sources while preserving the data can greatly help accessbility.

This guide borrows from Huggingface's (🤗) formal guide to creating a [shared dataset](https://huggingface.co/docs/datasets/share_dataset.html) and [adding a dataset](https://huggingface.co/docs/datasets/add_dataset.html). 

For simplicitly, this guide will step-by-step walk you through how to implement two types of biomedical datasets.

**Option 1. Public license data**<br>
**Option 2. Externally licensed data**

### Pre-requisites

Please make a github account prior to implementing a dataset; you can follow instructions to install git [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git). You will also need at Python 3.6+.

### 1. **Assign yourself a dataset**

Please select an issue on the biomedical data science repository [here](https://github.com/bigscience-workshop/biomedical/issues). Ideally, also check to see if the dataset already exists in the 🤗 [Hub](https://huggingface.co/datasets). 

If the dataset does not exist OR if an issue is open (there are no people assigned to the issue) please feel free to assign yourself to the issue. To do so, make a comment on the issue that you would like to take this dataset, and an organizer will add you to the repo.

### 2. **Setup a local version of the datasets repo to update**
Fork the dataset [repository](https://github.com/huggingface/datasets) from huggingface to your local github account. To do this, click the link to the repository and click "fork" in the upper-right corner. You should get an option to fork to your account, provided you are signed into Github. 

After you fork, clone the repository locally. You can do so as follows:

    git clone git@github.com:<your_github_username>/datasets.git
    cd datasets  # enter the directory

Next, you want to set your `upstream` location to enable you to push/pull (add or receive updates). You can do so as follows:
    
    git remote add upstream https://github.com/huggingface/datasets.git

You can optionally check that this was set properly by running the following command:
    
    git remote -v 

The output of this command should look as follows:

    origin  https://github.com/<your_github_username>/datasets (fetch)
    origin  https://github.com/<your_github_username>/datasets (push)
    upstream    https://github.com/huggingface/datasets.git (fetch)
    upstream    https://github.com/huggingface/datasets.git (push)

If you do NOT have an `origin` for whatever reason, then run:

    git remote add origin https://github.com/<your_github_username>/datasets.git

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

### 3. **Create a development environment** 
You can make an environment in any way you choose to. We highlight two possible options:

#### 3a) Create a conda environment

- Install [anaconda](https://docs.anaconda.com/anaconda/install/) for your appropriate operating system.
- Run the following command while in the `datasets` folder (you can pick your python version):

```
conda create --name <your_env_name_here> python=<your_python_version_here>  # Creates a conda env
conda activate <your_env_name_here> # Activate your conda environment
pip install -e ".[dev]"  # Install this while in the datasets folder
```

Make sure you are using the correct pip by checking `which pip`; this should point to the pip within your conda environment. You can deactivate your environment at any time by either exiting your terminal or using `conda deactivate`.

#### 3b) Create a venv environment

Python 3.3+ has venv automatically installed; official information is found [here](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).

```
python3 -m venv <your_env_name_here>
source <your_env_name_here>/bin/activate  # activate environment
pip install -e ".[dev]"  # Install this while in the datasets folder
```

**Note! If you already have the `datasets` library installed in a different environment of choice, please uninstall it first (via `pip uninstall datasets`) and re-install in the editable mode**

### 4. Implement your dataset

Make a new directory within the `datasets` folder as such: <br>

    mkdir datasets/<name_of_my_dataset>
    cd datasets/<name_of_my_dataset>

To implement your dataset, follow instructions in the necessary files:

- **Option 1:** To add a dataset that has a public license, use `template.py`
- **Option 2:** To add a dataset where files are local, use `template_local.py`

Make sure your dataset is implemented correctly by checking in python the following commands:

```python
import datasets
from datasets import load_dataset

data = load_dataset('your_dataset_name')
```
### 5. Format code 

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

