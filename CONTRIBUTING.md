# Guide to Implementing a dataset

The following guide will walk you through how to make a data-loading script for your dataset.

For the first step, it's important to create a **reproducible test environment**. To do so, please make a python environment with the following instructions:

## 1) Create a development environment
You can make an environment in any way you choose to. We highlight two possible options:

#### 1a) Create a conda environment

- Install [anaconda](https://docs.anaconda.com/anaconda/install/) for your appropriate operating system.
- Run the following command while in the `datasets` folder (you can pick your python version):

```
conda create --name <your_env_name_here> python=<your_python_version_here>  # Creates a conda env
conda activate <your_env_name_here> # Activate your conda environment
```

Make sure you are using the correct pip by checking `which pip`; this should point to the pip within your conda environment. You can deactivate your environment at any time by either exiting your terminal or using `conda deactivate`.

#### 1b) Create a venv environment

Python 3.3+ has venv automatically installed; official information is found [here](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).

```
python3 -m venv <your_env_name_here>
source <your_env_name_here>/bin/activate  # activate environment
```

## 2) Activate the Huggingface hub

You can find the official instructions [here](https://huggingface.co/welcome). We will provide what you need for the biomedical-datasets hackathon environment.

Download the `requirements.txt` file from [bigscience-biomedical](https://github.com/bigscience-workshop/biomedical/blob/master/requirements.txt).

With your environment active, install the requirements with `pip install -r requirements.txt`

<!-- @NATASHA tidy up requirements.txt -->

With the requirements downloaded, run the following command:

```
huggingface-cli login
```

Login with your 🤗 Hub account username and password. 

## 3) Implementation details

TBA

### 4) Format code 

Run the following commands:

    make style
    make quality

This runs the black formatter, isort, and lints to ensure that the code is readable and looks nice. Flake8 linting errors may require manual changes.

<!---
@NATASHA adapt make style/quality
-->

## 5) Test your data-loader 

Run the following command **in a folder that does not include your data-loading script**:

```python
from datasets import load_dataset

dataset = load_dataset("bigscience-biomedical/<your_dataset_name>", use_auth_token=True)
```