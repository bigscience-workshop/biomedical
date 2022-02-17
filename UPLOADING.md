# Uploading a dataloader script to the Hub

### 1. Make an account on the Hub

Please do the following before getting started: 

- [Make](https://huggingface.co/join) an account on 🤗's Hub and [login](https://huggingface.co/login). **Choose a good password, as you'll need to authenticate your credentials**. 

- Join the BigScience Biomedical initiative [here](https://huggingface.co/bigscience-biomedical)
    - click the "Request to join this org" button in the upper right corner.

- Make a github account; you can follow instructions to install git [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git). 


**Note - your permissions will be set to READ. Please contact an admin in a github issue to be granted WRITE access**

### 2) Activate the Huggingface hub

You can find the official instructions [here](https://huggingface.co/welcome). We will provide what you need for the biomedical-datasets hackathon environment.

Download the `requirements.txt` file from [bigscience-biomedical](https://github.com/bigscience-workshop/biomedical/blob/master/requirements.txt).

With your environment active, install the requirements with `pip install -r requirements.txt`

<!-- @NATASHA tidy up requirements.txt -->

With the requirements downloaded, run the following command:

```
huggingface-cli login
```

Login with your 🤗 Hub token generated from https://huggingface.co/settings/token. 

### 3. Create a dataset repository

Make a repository via the 🤗 Hub [here](https://huggingface.co/new-dataset) with the following details

+ Set Owner: bigscience-biomedical
+ Set Dataset name: the name of the dataset 
+ Select Private
+ Click `Create dataset`

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

At this point, there should be **no further changes in your code**. 

## 6) Test your data-loader 

Run the following command **in a folder that does not include your data-loading script**:

```python
from datasets import load_dataset

dataset = load_dataset("bigscience-biomedical/<your_dataset_name>", use_auth_token=True)
```

And with that, you have successfully contributed a data-loading script! 
