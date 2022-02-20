***Update 2022.02.14: We're close to launch! We're excited to have you!***

# Welcome to the BigScience🌸 Biomedical NLP Hackathon!

Huggingface's BigScience🌸 initative is an open scientific collaboration of nearly 600 researchers from 50 countries and 250 institutions who collaborate on various projects within the natural language processing (NLP) space to broaden accessibility of language datasets while working on challenging scientific questions around language modeling.  
<!--- @Natasha From the Data_sourcing wiki  --->

We are running a **Biomedical Datasets hackathon** to centralize many NLP datasets in the biological and medical space. Biological data is often diverse, so a unified location that joins multiple sources while preserving the data closest to the original form can greatly help accessbility.

## Goals of this hackathon

Our goal is to **enable easy programatic access to these datasets** using 🤗's [`datasets` library](https://huggingface.co/docs/datasets/). To do this, we propose a unified schema for dataset extraction, with the intention of implementing as many biomedical datasets as possible to enable **reproducibility in data processing**. 

We are leveraging Huggingface's (🤗) **Community Hub** in order to centralize these scripts so that practioners and researchers have easy access to these tools with a simple API.

There are two broad categories of biomedical datasets:

##### 1. Publically licensed data
##### 2. Externally licensed data

We will accept data-loading scripts for either type; please see the [FAQs](#FAQs) for more explicit details on what we propose.


## Contribution Guidelines

To be guaranteed acknowledgement, participants must implement an *accepted data-loading script* to the bigscience-biomedical collection for **at least 1 dataset**. Explicit instructions are found in [Get started](#Get-started), but the overall criteria to get accepted is as follows: <br>

- Implement a data-loading script as a branch
- PR the branch to the bigscience-biomedical repo
- If an admin approves the PR, follow the instructions on [uploading to the hub](UPLOADING.md).

Details for contributor acknowledgements and rewards can be found [here](#Thank-you)

*Optional* If you are looking for the official guides for contributing to the `datasets` library from Huggingface for a [shared dataset](https://huggingface.co/docs/datasets/share_dataset.html) or to [add a dataset](https://huggingface.co/docs/datasets/add_dataset.html), you can find them in the links attached. Our guide follows closely from these, with adaptations to suit this intitiative.

## Get started!

Please make a github account prior to implementing a dataset; you can follow instructions to install git [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git). 

You will also need at Python 3.6+. If you are installing python, we recommend downloading [anaconda](https://docs.anaconda.com/anaconda/install/index.html) to curate a python environment with necessary packages. 

All commands in the guide provided are done through terminal access.

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

If an admin approves it, then you are welcome to implement this dataset and it will count toward contribution credit.

### 2. Implement the dataloader for your dataset

A step-by-step guide on how you can implementing a data-loading script can be found [here](CONTRIBUTING.md).

Please ensure your dataloader follows our expected biomedical schema, found [here](#Template-scripts)

**Please do not upload the data directly; if you have a specific question or request, [reach out to an admin](#Community-channels)**

#### Template scripts

We provide template scripts for the most common biomedical NLP schemas: <br>

1. [Named entity recognition (NER), relational extraction (RE), event extraction](templates/template_ner.py)
2. [Translation](templates/template_translation.py)
3. [Part-of-speech (POS)](templates/template_pos.py)
4. [Question-answering (QA)](templates/template_qa.py)
5. [Natural language inference (NLI)](templates/template_nli.py)
6. [Normalization](templates/template_normalization.py)
7. [Sentence classification](templates/template_sentcls.py)
8. [Fact verification](templates/template_factver.py)
9. [Semantic similarity](templates/template_semsim.py)
10. [Topic classification](templates/template_topiccls.py)
11. [Co-reference](templates/template_coref.py)

**Don't see your task or have questions? [Reach out to an admin](#Community-channels)**

We have provided some examples of how to implement a dataset as follows:

#### Example scripts
You can find examples scripts on how to implement a dataset here: <br>

1. [Template for publically licensed data](templates/template.py)
2. [Template for externally licensed data](templates/template_local.py)
3. [Example of publically licensed data](examples/chemprot.py)
4. [Example of externally licensed data](examples/cellfinder.py)
5. [Example of Bio-C format annotation](examples/bc5cdr.py)

### 3. PR your dataloader!

Explicit instructions on how to PR a dataloader are found [here](CONTRIBUTING.md). Once you do, an admin will code-review your changes. Admins may propose changes before acceptance, or accept as-is. Please feel free to reach out to get your PRs accepted!

Once the PR is accepted, please follow the instructions to upload the dataset into the [Hub](UPLOADING.md).

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

We understand that some biomedical datasets require external licensing. To respect the agreement of the license, we recommend implementing a dataloader script that works if the user has the dataset file(s) stored locally. You can find an example [here](examples/cellfinder.py).

*What types of libraries can we import?*

Eventually, your dataloader script will need to run using only the packages supplied by the [datasets](https://github.com/huggingface/datasets) package. If you find a well supported package that makes your implementation easier (e.g. [bioc](https://github.com/bionlplab/bioc)), then feel free to use it. We will address the specifics during review of your PR to the [BigScience biomedical repo](https://github.com/bigscience-workshop/biomedical) and find a way to make it usable in the final submission to [huggingface bigscience-biomedical](https://huggingface.co/bigscience-biomedical)

*Can I upload the dataset directly?*

No. Please do not upload your dataset directly. This is not the goal of the hackathon and many datasets have external licensing agreements. If the dataset is public (i.e. can be downloaded without credentials or signed data user agreement), include a downloading component in your dataset loader script. Otherwise, include only an "extraction from local files" component in your dataset loader script. You can see examples of both in the [examples](https://github.com/bigscience-workshop/biomedical/tree/master/examples) directory. If you have a custom dataset you would like to submit, please [make an issue](https://github.com/bigscience-workshop/biomedical/issues/new) and an admin will get back to you.  

*My dataset supports multiple tasks with different bigbio schemas. What should I do? 

In some cases, a single dataset will support multiple tasks with different bigbio schemas. For example, the `muchmore` dataset can be used for a translation task (supported by the `text_to_text` schema) and a named entity recognition task (supported by the `kb` schema). In this case, please implement one config for each task and name the config `bigbio-<task>`. In the `muchmore` example, this would mean one config called `bigbio-translation` and one config called `bigbio-ner`.  

*My dataset is complicated, can you help me?*

Yes! Please join the hack-a-thon [Biomedical Discord Server](https://discord.gg/PrhGdhJE) and ask for help. 

*My dataset is too complicated, can I switch?*

Yes! Some datasets are easier to write dataloader scripts for than others. If you find yourself working on a dataset that you can not make progress on, please make a comment in the associated issue, asked to be un-assigned from the issue, and start the search for a new unclaimed dataset. 

## Thank you!

We greatly appreciate your help - as a token or our gratitude, contributors can get the following rewards:

* Authorship on a forthcoming paper focusing on the construction of biomedical datasets (requires the contribution of 1 full dataset)
* Recognition of your work on both the BigScience biomedical github repo and on an a forthcoming landing page 


The hackathon guide is heavily inspired from [here](https://github.com/bigscience-workshop/data_tooling/wiki/datasets-hackathon).

<!---
@Natasha
Contribution rewards:

- t-shirts?
- can we get a github star/badge that people can host on their profiles
- minimum acknowledgement in a paper; may have authorship
-->
