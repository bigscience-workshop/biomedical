## Uploading a dataloader script to the Hub

### 1. Make an account on the Hub

Please do the following before getting started: 

- [Make](https://huggingface.co/join) an account on 🤗's Hub and [login](https://huggingface.co/login). **Choose a good password, as you'll need to authenticate your credentials**. 

- Join the BigScience Biomedical initiative [here](https://huggingface.co/bigscience-biomedical)
    - click the "Request to join this org" button in the upper right corner.

- Make a github account; you can follow instructions to install git [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git). 


**Note - your permissions will be set to READ. Please contact an admin in a github issue to be granted WRITE access**

### 2. Create a dataset repository

Make a repository via the 🤗 Hub [here](https://huggingface.co/new-dataset) with the following details

+ Set Owner: bigscience-biomedical
+ Set Dataset name: the name of the dataset 
+ Select Private
+ Click `Create dataset`

### 3. Clone the dataset repository

Using terminal access, find a location to place your github repository. In this location, use the following command:

```
git clone https://huggingface.co/datasets/bigscience-biomedical/<your_dataset_name>
```

### 4. Commit your changes

Run the following commands to add and push your work

```
git add <your_file_name.py>  # add the dataset
git commit -m "Adds <your_dataset_name>"
git push origin
```

And with that, you have successfully contributed a data-loading script! You can check out your script as follows:

```python
from datasets import load_dataset
data = load_dataset("bigscience-biomedical/<your_dataset_name>")
```