# Uploading a dataloader script to the Hub

**At this point, there should be no further changes to your dataloader script after the PR was accepted**.

### 1. Make an account on the Hub

Please do the following before getting started: 

- [Make](https://huggingface.co/join) an account on 🤗's Hub and [login](https://huggingface.co/login). **Choose a good password, as you'll need to authenticate your credentials**. 

- Join the BigScience Biomedical initiative [here](https://huggingface.co/bigscience-biomedical)
    - click the "Request to join this org" button in the upper right corner.

- Make a github account; you can follow instructions to install git [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git). 


**Note - your permissions will be set to READ. Please contact an admin in your dataset's github issue to be granted WRITE access; this should be given after your PR is accepted**.

### 2) Activate the Huggingface hub

You can find the official instructions [here](https://huggingface.co/welcome). We will provide what you need for the biomedical-datasets hackathon environment.

With your active `bigscience-biomedical` environment, use the following command:

```
huggingface-cli login
```

Login with your 🤗 Hub account username and password. 

### 3. Create a dataset repository

Make a repository via the 🤗 Hub [here](https://huggingface.co/new-dataset) with the following details

+ Set Owner: bigscience-biomedical
+ Set Dataset name: the name of the dataset 
+ Set License: the license that applies to this dataset
+ Select Private
+ Click `Create dataset`

**Please name your dataloading script with the same name as the dataset.** For example, if your dataset loader script is called `n2c2_2011_coref.py`, then your dataset name should be `n2c2_2011_coref`.

If there is no appropriate license available in the provided options (for example for datasets with specific data user agreements) you should select "other". 

### 4. Clone the dataset repository

Using terminal access, find a location to place your github repository. In this location, use the following command:

```
git clone https://huggingface.co/datasets/bigscience-biomedical/<your_dataset_name>
```

### 5. Commit your changes

Run the following commands to add and push your work

```
git add <your_file_name.py>  # add the dataset
git commit -m "Adds <your_dataset_name>"
git push origin
```

## 6) Test your data-loader 

Run the following command **in a folder that does not include your data-loading script**:

Test both the original dataset schema/config and the bigbio schema/config. 

**Public Dataset**
```python
from datasets import load_dataset

dataset_orig = load_dataset("bigscience-biomedical/<your_dataset_name>", name="source", use_auth_token=True)
dataset_bigbio = load_dataset("bigscience-biomedical/<your_dataset_name>", name="bigbio", use_auth_token=True)
```

**Private Dataset**

```python
from datasets import load_dataset

dataset_orig = load_dataset(
    "bigscience-biomedical/<your_dataset_name>", 
    name="source", 
    data_dir="/local/path/to/data/files",
    use_auth_token=True)
dataset_bigbio = load_dataset(
    "bigscience-biomedical/<your_dataset_name>", 
    name="bigbio", 
    data_dir="/local/path/to/data/files",
    use_auth_token=True)
```

And with that, you have successfully contributed a data-loading script! 
