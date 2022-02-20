***Update 2022.02.21: We're close to launch! We're excited to have you!***

# Welcome to the BigScience🌸 Biomedical NLP Hackathon!

Huggingface's BigScience🌸 initative is an open scientific collaboration of nearly 600 researchers from 50 countries and 250 institutions who collaborate on various projects within the natural language processing (NLP) space to broaden accessibility of language datasets while working on challenging scientific questions around language modeling.  
<!--- @Natasha From the Data_sourcing wiki  --->

We are running a **Biomedical Datasets hackathon** to centralize many NLP datasets in the biological and medical space. Biological data is often diverse, so a unified location that joins multiple sources while preserving the data closest to the original form can greatly help accessbility.

## Goals of this hackathon

Our goal is to **enable easy programatic access to these datasets** using Huggingface's (🤗) [`datasets` library](https://huggingface.co/docs/datasets/). To do this, we propose a unified schema for dataset extraction, with the intention of implementing as many biomedical datasets as possible to enable **reproducibility in data processing**. 

We are leveraging the 🤗 **Community Hub** in order to centralize these scripts so that practioners and researchers have easy access to these tools with a simple API.

There are two broad categories of biomedical datasets:

##### 1. Publically licensed data
##### 2. Externally licensed data

We will accept data-loading scripts for either type; please see the [FAQs](#FAQs) for more explicit details on what we propose.

### Why is this important?

Biomedical language data is highly specialized, requiring expert curation and annotation. Many great initiatives have created different language data sets across a variety of biological domains. A **centralized source that allows users to access relevant information reproducibly** greatly increases accessibility of these datasets, and promotes research.

Our unified schema allows researchers and practioners to **access the same type of information across a variety of datasets with fixed keys**. This can enable researchers to quickly iterate, and write scripts without worrying about pre-processing nuances specific to a dataset.


## Contribution Guidelines

To be considered a contributor, participants must implement an *accepted data-loading script* to the bigscience-biomedical collection for **at least 1 dataset**. 

Explicit instructions are found in [Get started](#Get-started), but the overall criteria to get accepted is as follows: <br>

- Write a data-loading script for a dataset in a new branch
- PR your branch to the `bigscience-biomedical` repo and ping the admins
- If an admin approves the PR, follow the instructions on [uploading to the hub](UPLOADING.md).

Details for contributor acknowledgements and rewards can be found [here](#Thank-you)

## Get started!

Please make a github account prior to implement a dataset; you can follow instructions to install git [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git). 

You will also need at least Python 3.6+. If you are installing python, we recommend downloading [anaconda](https://docs.anaconda.com/anaconda/install/index.html) to curate a python environment with necessary packages. **We strongly recommend Python 3.8+ for stability**. 

All commands in the guide provided are done through terminal access. If you need help, please [reach out to an admin](#Community-channels).

### 1. Choose a dataset to implement

There are two options to choose a dataset to implement; you can choose either option, but **we recommend option A**. 

**Option A: Assign yourself a dataset from our curated list**
- Choose a dataset from the [list of Biomedical datasets](https://github.com/orgs/bigscience-workshop/projects/6/). 

- Assign yourself an issue by clicking the dataset in the project list, and comment `#self-assign` under the issue. **Please assign yourself to issues with no other collaborators assigned**. You should see your GitHub username associated to the issue within 1-2 minutes of making a comment.

- Search to see if the dataset exists in the 🤗 [Hub](https://huggingface.co/datasets). If it exists, please comment on the original issue with the link and choose another dataset to implement.

**Option B: Implement a new dataset not on the list**

If you have a biomedical or clinical dataset you would like to propose in this collection, you are welcome to [make a new issue](https://github.com/bigscience-workshop/biomedical/issues/new). Choose `Add Dataset` and fill out relevant information. **Make sure that your dataset does not exist in the 🤗 [Hub](https://huggingface.co/datasets).**

If an admin approves it, then you are welcome to implement this dataset and it will count toward contribution credit.

### 2. Implement the dataloader for your dataset

A step-by-step guide on how you can implement a data-loading script can be found [here](CONTRIBUTING.md).

**Please do not upload the data directly; if you have a specific question or request, [reach out to an admin](#Community-channels)**

Please ensure your dataloader follows our expected biomedical schema, found [here](#task_schemas.md)

### 3. Make a pull-request (PR) for your dataloader!

Before your data-loading script is accepted, you will need to make a PR to the big-science biomedical repo. Explicit instructions on how to PR a dataloader are found [here](CONTRIBUTING.md). 

Once you do, an admin will code-review your changes. Admins may propose changes before acceptance, or accept as-is. Please feel free to reach out to get your PRs accepted!

Once the PR is accepted, please follow the instructions to upload the dataset into the [Hub](UPLOADING.md).

## Community channels

We welcome contributions from a wide variety of backgrounds; we are more than happy to guide you through the process. For instructions on how to get involved or ask for help, check out the following options:

#### Join BigScience
Please join the BigScience initiative [here](https://bigscience.huggingface.co/); there is a [google form](https://docs.google.com/forms/d/e/1FAIpQLSdF68oPkylNhwrnyrdctdcs0831OULetgfYtr-aVxBg053zqA/viewform) to fill out to have access to the biomedical working group slack. Once you have filled out this form, you'll get access to BigScience's google drive. There is a document where you can fill your name next to a working group; be sure to fill your name on the "Biomedical" group. 

#### Join our Discord Server
Alternatively, you can ping us on the [Biomedical Discord Server](https://discord.gg/Cwf3nT3ajP). The Discord server can be used to share information quickly or ask code-related questions.

#### Make a Github Issue
For quick questions and clarifications, you can [make an issue via Github](https://github.com/bigscience-workshop/biomedical/issues).

You are welcome to use any of the above resources as necessary. 

## FAQs

##### What if my dataset does not have a public license?

We understand that some biomedical datasets require external licensing. To respect the agreement of the license, we recommend implementing a dataloader script that works if the user has a locally downloaded file. You can find an example [here](examples/cellfinder.py) and follow the local [template](templates/template_local.py).

##### What if my dataset does not have a public license?

We understand that some biomedical datasets require external licensing. To respect the agreement of the license, we recommend implementing a dataloader script that works if the user has the dataset file(s) stored locally. You can find an example [here](examples/cellfinder.py).

##### What types of libraries can we import?

Eventually, your dataloader script will need to run using only the packages supplied by the [datasets](https://github.com/huggingface/datasets) package. If you find a well supported package that makes your implementation easier (e.g. [bioc](https://github.com/bionlplab/bioc)), then feel free to use it. We will address the specifics during review of your PR to the [BigScience biomedical repo](https://github.com/bigscience-workshop/biomedical) and find a way to make it usable in the final submission to [huggingface bigscience-biomedical](https://huggingface.co/bigscience-biomedical)

##### Can I upload the dataset directly?

No. Please do not upload your dataset directly. This is not the goal of the hackathon and many datasets have external licensing agreements. If the dataset is public (i.e. can be downloaded without credentials or signed data user agreement), include a downloading component in your dataset loader script. Otherwise, include only an "extraction from local files" component in your dataset loader script. You can see examples of both in the [examples](https://github.com/bigscience-workshop/biomedical/tree/master/examples) directory. If you have a custom dataset you would like to submit, please [make an issue](https://github.com/bigscience-workshop/biomedical/issues/new) and an admin will get back to you.  

##### My dataset supports multiple tasks with different bigbio schemas. What should I do? 

In some cases, a single dataset will support multiple tasks with different bigbio schemas. For example, the `muchmore` dataset can be used for a translation task (supported by the `text_to_text` schema) and a named entity recognition task (supported by the `kb` schema). In this case, please implement one config for each task and name the config `bigbio-<task>`. In the `muchmore` example, this would mean one config called `bigbio-translation` and one config called `bigbio-ner`.  

##### How should I handle offsets and text in the bigbio kb schema?

Full details on how to handle offsets and text in the bigbio kb schema can be found in the [schema documentation](https://github.com/bigscience-workshop/biomedical/blob/master/task_schemas.md).

##### My dataset is complicated, can you help me?

Yes! Please join the hack-a-thon [Biomedical Discord Server](https://discord.gg/PrhGdhJE) and ask for help. 

##### My dataset is too complicated, can I switch?

Yes! Some datasets are easier to write dataloader scripts for than others. If you find yourself working on a dataset that you can not make progress on, please make a comment in the associated issue, asked to be un-assigned from the issue, and start the search for a new unclaimed dataset. 

## Thank you!

We greatly appreciate your help - as a token or our gratitude, contributors can get the following rewards:

* **Authorship on a paper**; we are submitting this work to various venues centered on programmatic access to biomedical literature

* **Recognition as an official contributor** if your script is accepted, you will be an official author on the **BigScience Biomedical library** from Huggingface.

The hackathon guide is heavily inspired from [here](https://github.com/bigscience-workshop/data_tooling/wiki/datasets-hackathon).
