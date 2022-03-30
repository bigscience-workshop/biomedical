***Update 2022.03.26: We're close to launch! Hackathon is scheduled for April 2nd - April 15th. We're excited to have you!***

# Welcome to the BigScience🌸 Biomedical NLP Hackathon!

Huggingface's BigScience🌸 initative is an open scientific collaboration of nearly 600 researchers from 50 countries and 250 institutions who collaborate on various projects within the natural language processing (NLP) space to broaden accessibility of language datasets while working on challenging scientific questions around language modeling.  
<!--- @Natasha From the Data_sourcing wiki  --->

We are running a **Biomedical Datasets hackathon** to centralize many NLP datasets in the biological and medical space. Biological data is diverse, so a unified location that joins multiple sources while preserving the data closest to the original form can greatly help accessbility.

## Goals of this hackathon

Our goal is to **enable easy programatic access to these datasets** using Huggingface's (🤗) [`datasets` library](https://huggingface.co/docs/datasets/). To do this, we propose a unified schema for dataset extraction, with the intention of implementing as many biomedical datasets as possible to enable **reproducibility in data processing**. 

There are two broad licensing categories for biomedical datasets:

##### 1. Public Data (Public Domain, Creative Commons, Apache 2.0, etc.)
##### 2. External Data Use Agreements (PhysioNet, i2b2/n2c2, etc.)

We will accept data-loading scripts for either type; please see the [FAQs](#FAQs) for more explicit details on what we propose.

### Why is this important?

Biomedical language data is highly specialized, requiring expert curation and annotation. Many great initiatives have created different language data sets across a variety of biological domains. A **centralized source that allows users to access relevant information reproducibly** greatly increases accessibility of these datasets, and promotes research.

Our unified schema allows researchers and practioners to **access the same type of information across a variety of datasets with fixed keys**. This can enable researchers to quickly iterate, and write scripts without worrying about pre-processing nuances specific to a dataset.


## Contribution Guidelines

To be considered a contributor, participants must implement an *accepted data-loading script* to the bigscience-biomedical collection for **at least 3 datasets**. 

Explicit instructions are found in the next section, but the steps for getting a data-loading script accepted are as follows: <br>

- Fork this repo and write a data-loading script in a new branch
- PR your branch back to this repo and ping the admins
- An admin will review and approve your PR or ping you for changes

Details for contributor acknowledgements and rewards can be found [here](#Thank-you)

## Get started!

### 1. Choose a dataset to implement

There are two options to choose a dataset to implement; you can choose either option, but **we recommend option A**. 

**Option A: Assign yourself a dataset from our curated list**
- Choose a dataset from the [list of Biomedical datasets](https://github.com/orgs/bigscience-workshop/projects/6/). 
<p align="center">
    <img src="./docs/_static/img/select-task.jpg" style="width: 75%;"/>
</p>

- Assign yourself an issue by clicking the dataset in the project list, and comment `#self-assign` under the issue. **Please assign yourself to issues with no other collaborators assigned**. You should see your GitHub username associated to the issue within 1-2 minutes of making a comment.
<p align="center">
    <img src="./docs/_static/img/self-assign.jpg" style="width: 75%;"/>
</p>

- Search to see if the dataset exists in the 🤗 [Hub](https://huggingface.co/datasets). If it exists, please use the current implementation as the `source` and focus on implementing the [task-specific `bigbio` schema](https://github.com/bigscience-workshop/biomedical/blob/master/task_schemas.md). 

**Option B: Implement a new dataset not on the list**

If you have a biomedical or clinical dataset you would like to propose in this collection, you are welcome to [make a new issue](https://github.com/bigscience-workshop/biomedical/issues/new/choose). Choose `Add Dataset` and fill out relevant information. **Make sure that your dataset does not exist in the 🤗 [Hub](https://huggingface.co/datasets).**

If an admin approves it, then you are welcome to implement this dataset and it will count toward contribution credit.

### 2. Implement the data-loading script for your dataset and create a PR

[Check out our step-by-step guide to implementing a dataloader with the bigbio schema](CONTRIBUTING.md).

**Please do not upload the data directly; if you have a specific question or request, [reach out to an admin](#Community-channels)**

## Community channels

We welcome contributions from a wide variety of backgrounds; we are more than happy to guide you through the process. For instructions on how to get involved or ask for help, check out the following options:

#### Join BigScience
Please join the BigScience initiative [here](https://bigscience.huggingface.co/); there is a [google form](https://docs.google.com/forms/d/e/1FAIpQLSdF68oPkylNhwrnyrdctdcs0831OULetgfYtr-aVxBg053zqA/viewform) to fill out to have access to the biomedical working group slack. Once you have filled out this form, you'll get access to BigScience's google drive. There is a document where you can fill your name next to a working group; be sure to fill your name on the "Biomedical" group. 

#### Join our Discord Server
Alternatively, you can ping us on the [Biomedical Discord Server](https://discord.gg/Cwf3nT3ajP). The Discord server can be used to share information quickly or ask code-related questions.

#### Make a Github Issue
For quick questions and clarifications, you can [make an issue via Github](https://github.com/bigscience-workshop/biomedical/issues/new/choose).

You are welcome to use any of the above resources as necessary. 

## FAQs

#### What if my dataset does not have a public license?

We understand that some biomedical datasets require external licensing. To respect the agreement of the license, we recommend implementing a dataloader script that works if the user has a locally downloaded file. You can find an example [here](examples/cellfinder.py) and follow the local/private dataset specific instructions in  [template](templates/template.py).

#### What types of libraries can we import?

Eventually, your dataloader script will need to run using only the packages supplied by the [datasets](https://github.com/huggingface/datasets) package. If you find a well supported package that makes your implementation easier (e.g. [bioc](https://github.com/bionlplab/bioc)), then feel free to use it. 

We will address the specifics during review of your PR to the [BigScience biomedical repo](https://github.com/bigscience-workshop/biomedical) and find a way to make it usable in the final submission to [huggingface bigscience-biomedical](https://huggingface.co/bigscience-biomedical)

#### Can I upload the dataset directly?

No. Please do not upload your dataset directly. This is not the goal of the hackathon and many datasets have external licensing agreements. If the dataset is public (i.e. can be downloaded without credentials or signed data user agreement), include a downloading component in your dataset loader script. Otherwise, include only an "extraction from local files" component in your dataset loader script. You can see examples of both in the [examples](https://github.com/bigscience-workshop/biomedical/tree/master/examples) directory. If you have a custom dataset you would like to submit, please [make an issue](https://github.com/bigscience-workshop/biomedical/issues/new) and an admin will get back to you.  

#### My dataset supports multiple tasks with different bigbio schemas. What should I do? 

In some cases, a single dataset will support multiple tasks with different bigbio schemas. For example, the `muchmore` dataset can be used for a translation task (supported by the `Text to Text (T2T)` schema) and a named entity recognition task (supported by the `Knowledge Base (KB)` schema). In this case, please implement one config for each supported schema and name the config `<datasetname>_bigbio_<schema>`. In the `muchmore` example, this would mean one config called `muchmore_bigbio_t2t` and one config called `muchmore_bigbio_kb`.  

#### How should I handle offsets and text in the bigbio schema?

Full details on how to handle offsets and text in the bigbio kb schema can be found in the [schema documentation](https://github.com/bigscience-workshop/biomedical/blob/master/task_schemas.md).

#### My dataset is complicated, can you help me?

Yes! Please join the hack-a-thon [Biomedical Discord Server](https://discord.gg/Cwf3nT3ajP) and ask for help. 

#### My dataset is too complicated, can I switch?

Yes! Some datasets are easier to write dataloader scripts for than others. If you find yourself working on a dataset that you can not make progress on, please make a comment in the associated issue, asked to be un-assigned from the issue, and start the search for a new unclaimed dataset. 

## Thank you!

We greatly appreciate your help! 

The artifacts of this hackathon will be described in a forthcoming academic paper targeting a machine learning or NLP audience. Implementing 3 or more dataset loaders will guarantee authorship. We recognize that some datasets require more effort than others, so please reach out if you have questions. Our goal is to be inclusive with credit!

## Acknowledgements

This hackathon guide was heavily inspired by [the BigScience Datasets Hackathon](https://github.com/bigscience-workshop/data_tooling/wiki/datasets-hackathon).
