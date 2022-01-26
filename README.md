***Update 2022.01.26: This is a WIP stay tuned for the launch of this hackathon! We're excited to have you!***

# Welcome to the BigScience🌸 Biomedical NLP Hackathon!

Huggingface's BigScience🌸 initative is an open scientific collaboration of nearly 600 researchers from 50 countries and 250 institutions who collaborate on various projects within the natural language processing (NLP) space to broaden accessibility of language datasets while working on challenging scientific questions around language modeling.  
<!--- @Natasha From the Data_sourcing wiki  --->

We are running a **Biomedical Datasets hackathon** to centralize many NLP datasets in the biological and medical space. Biological data is often diverse, so a unified location that joins multiple sources while preserving the data closest to the original form can greatly help accessbility.

## Goals of this hackathon

Our goal is to **enable easy programatic access to these datasets** using 🤗's [`datasets` library](https://huggingface.co/docs/datasets/). To do this, we propose a unified schema for dataset extraction, with the intention of implementing as many biomedical datasets as possible to enable **reproducibility in data processing**. We are leveraging Huggingface's (🤗) **Community Hub** in order to centralize these scripts so that practioners and researchers have easy access to these tools with a simple API.

There are two broad categories of biomedical datasets:

##### 1. Publically licensed data
##### 2. Externally licensed data

We will accept data-loading scripts for either type; please see the [FAQs](#FAQs) for more explicit details on what we propose.


### Scope and Future Vision
<!---
Here, we should write maybe 1-3 sentences around our plans for prompting.
-->

## Contribution Guidelines

There are official guides to contributing to the `datasets` library from Huggingface's  for a [shared dataset](https://huggingface.co/docs/datasets/share_dataset.html) and to [add a dataset](https://huggingface.co/docs/datasets/add_dataset.html). Our guide follows closely from these, with adaptations to suit this intitiative.

Contributors must implement an *accepted data-loading script* to the collection for **at least 1 dataset** to be guaranteed acknowledgement. All PRs submitted will be subject to code review prior to acceptance.

Details for contributor acknowledgements and rewards can be found [here](#Thank-you)

## Get started

### Pre-Requisites

Please do the following before getting started: 

- [Make](https://huggingface.co/join) an account on 🤗's Hub and [login](https://huggingface.co/login). **Choose a good password, as you'll need to authenticate your credentials**. 

- Join the BigScience Biomedical initiative [here](https://huggingface.co/bigscience-biomedical)
    - click the "Request to join this org" button in the upper right corner.

- Make a github account; you can follow instructions to install git [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git). 

*Optional* <br>
- You will also likely need at Python 3.6+. If you are installing python, we recommend downloading [anaconda](https://docs.anaconda.com/anaconda/install/index.html) to curate a python environment with necessary packages. 

### 1. Choose a dataset to implement

There are two options to choose a dataset to implement; you can choose either option, but **we recommend option A**. 

**Option A: Assign yourself a dataset from our curated list**
- Choose a dataset from the [list of Biomedical datasets](https://github.com/orgs/bigscience-workshop/projects/6/). 

- If there are no volunteers assigned to the dataset, go to the issue and assign yourself to the issue by commenting on it. An organizer will formally assign you to the issue with your github account handle.

<!-- @NATASHA TODO create #self-assign Github actions -->

- Search to see if the dataset exists in the 🤗 [Hub](https://huggingface.co/datasets). If it exists, please comment on the original issue with the link and choose another dataset to implement.

**Option B: Implement a new dataset not on the list**

If you have a biomedical or clinical dataset you would like to propose in this collection, you are welcome to [make a new issue](https://github.com/bigscience-workshop/biomedical/issues/new). 

In your new issue, please provide the following details:

- Task (Ex: Named Entity Recognition "NER")
- License (Ex: MIT License)
- Format (Ex: BioC/BRAT)
- Language (Ex: English)
- Citation or Source

Ideally, also check to see if the dataset already exists in the 🤗 [Hub](https://huggingface.co/datasets). 

If an admin approves it, the dataset will count toward the hackathon contributions.

### 2. Create a dataset repository

Make a repository via the 🤗 Hub [here](https://huggingface.co/new-dataset) with the following details

+ Set Owner: bigscience-biomedical
+ Set Dataset name: the name of the dataset 
+ Select Private
+ Click `Create dataset`

### 3. Clone the dataset repository

Using terminal access, find a location to place your github repository. In this location, use the following command:

```
git clone https://huggingface.co/datasets/bigscience-biomedical/chemprot
```
### 4. Implement the dataloader for your dataset

A step-by-step guide on how you can implementing a data-loading script can be found [here](CONTRIBUTING.md).

Please ensure your dataloader follows our expected biomedical schema <!---  @Natasha This needs to be a hyperlink [Biomedical Schema]() -->

**Please do not upload the data directly; if you have a specific question or request, [reach out to an admin](#Community-channels)**

Below, you can find some example template scripts

#### Template scripts

You can find template scripts and examples as follows: <br>

1. [Template for publically licensed data](templates/template.py)
2. [Template for externally licensed data](templates/template_local.py)
3. [Example of publically licensed data](examples/chemprot.py)
4. [Example of externally licensed data](examples/cellfinder.py)
5. [Example of Bio-C format annotation](examples/bc5cdr.py)
6. Example with BRAT format annotation (coming soon)

<!---
@NATASHA Make DDI script

[Example with BRAT format annotation](examples/ddi.py)
-->

### 5. Commit your changes

Run the following commands to add and push your work

```
git add <your_file_name.py>  # add the dataset
git commit -m "Adds <your_dataset_name"
git push origin
```

And with that, you have successfully contributed a data-loading script!

## Community channels

We welcome contributions from a wide variety of backgrounds; we are more than happy to guide you through the process. For instructions on how to get involved or ask for help, check out the following options:

#### Join BigScience
Please join the BigScience initiative [here](https://bigscience.huggingface.co/); there is a [google form](https://docs.google.com/forms/d/e/1FAIpQLSdF68oPkylNhwrnyrdctdcs0831OULetgfYtr-aVxBg053zqA/viewform) to fill out to have access to the biomedical working group slack. Once you have filled out this form, you'll get access to BigScience's google drive. There is a document where you can fill your name next to a working group; be sure to fill your name on the "Biomedical" group. 

#### Join our Discord Server
Alternatively, you can ping us on the [Biomedical Discord Server](https://discord.gg/PrhGdhJE). The Discord server can be used to share information quickly or ask code-related questions.

#### Make a Github Issue
For quick questions and clarifications, you can [make an issue via Github](https://github.com/bigscience-workshop/biomedical/issues).

You are welcome to use any of the above resources as necessary. 

## FAQs

*What if my dataset does not have a public license?*

We understand that some biomedical datasets require external licensing. To respect the agreement of the license, we recommend implementing a dataloader script that works if the user has a locally downloaded file. You can find an example [here](examples/cellfinder.py) and follow the local [template](templates/template_local.py).

*What types of libraries can we import?*

*Can I upload the data directly?*

*My dataset is complicated, can you help me?*

## Thank you!

We greatly appreciate your help - as a token or our gratitude, contributors can get the following rewards:

The hackathon guide is heavily inspired from [here](https://github.com/bigscience-workshop/data_tooling/wiki/datasets-hackathon).

<!---
@Natasha
Contribution rewards:

- t-shirts?
- can we get a github star/badge that people can host on their profiles
- minimum acknowledgement in a paper; may have authorship
-->