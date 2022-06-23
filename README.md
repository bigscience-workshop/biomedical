# BigBIO: Biomedical Datasets

BigBIO (BigScience Biomedical) is an open library of biomedical dataloaders built using Huggingface's (🤗) [`datasets` library](https://huggingface.co/docs/datasets/) for data-centric machine learning. Our goals include

- Lightweight, programmatic access to biomedical datasets at scale
- Promote reproducibility in data processing
- Better documentation for dataset provenance, licensing, and other key attributes
- Easier generation of meta-datasets (e.g., prompting, masssive MTL)

Currently BigBIO provides support for

- more than 120 biomedical datasets
- 10 languages
- Harmonized dataset schemas by task type
- Metadata on *licensing*, *coarse/fine-grained task types*, *domain*, and more!


### Documentation

- Tutorials: *Coming Soon!*
- [Volunteer Project Board](https://github.com/orgs/bigscience-workshop/projects/6): Implement or suggest new datasets
- [Contributor Guide](CONTRIBUTING.md)
- [Task Schema Overview](task_schemas.md)


## Installation

just want to use the package? pip install from github
```bash
pip install git+https://github.com/bigscience-workshop/biomedical.git
```

want to develop? pip install from cloned repo
```bash
git clone git@github.com:bigscience-workshop/biomedical.git
cd biomedical
pip install -e .
```


## Quick Start

### Initialize

Start by creating an instance of `BigBioConfigHelpers`.
This will help locate and filter datasets available in the bigbio package.

```python
from bigbio.dataloader import BigBioConfigHelpers
conhelps = BigBioConfigHelpers()
print("found {} dataset configs from {} datasets".format(
    len(conhelps),
    len(conhelps.available_dataset_names)
))
```

### BigBioConfigHelper

Each bigbio dataset has at least one source config and one bigbio config.
Source configs attempt to preserve the original structure of the dataset
while bigbio configs are normalized into one of several bigbio
[task schemas](task_schemas.md)

The `conhelps` container has one element for every config of every dataset.
The config helper for the source config of the
[BioCreative V Chemical-Disease Relation](https://biocreative.bioinformatics.udel.edu/tasks/biocreative-v/track-3-cdr/)
dataset looks like this,

```python
BigBioConfigHelper(
    script='/home/galtay/repos/biomedical/bigbio/biodatasets/bc5cdr/bc5cdr.py',
    dataset_name='bc5cdr',
    tasks=[
        <Tasks.NAMED_ENTITY_RECOGNITION: 'NER'>,
	<Tasks.NAMED_ENTITY_DISAMBIGUATION: 'NED'>,
	<Tasks.RELATION_EXTRACTION: 'RE'>
    ],
    languages=[<Lang.EN: 'English'>],
    config=BigBioConfig(
        name='bc5cdr_source',
        version=1.5.16,
        data_dir=None,
        data_files=None,
        description='BC5CDR source schema',
        schema='source',
        subset_id='bc5cdr'
    ),
    is_local=False,
    is_bigbio_schema=False,
    bigbio_schema_caps=None,
    is_large=False,
    is_resource=False,
    is_default=True,
    is_broken=False,
    bigbio_version='1.0.0',
    source_version='01.05.16',
    citation='@article{DBLP:journals/biodb/LiSJSWLDMWL16,\n  author    = {Jiao Li and\n               Yueping Sun and\n
Robin J. Johnson and\n               Daniela Sciaky and\n               Chih{-}Hsuan Wei and\n               Robert Leaman and\n
Allan Peter Davis and\n               Carolyn J. Mattingly and\n               Thomas C. Wiegers and\n               Zhiyong Lu},\n
title     = {BioCreative {V} {CDR} task corpus: a resource for chemical disease\n               relation extraction},\n  journal   =
{Database J. Biol. Databases Curation},\n  volume    = {2016},\n  year      = {2016},\n  url       =
{https://doi.org/10.1093/database/baw068},\n  doi       = {10.1093/database/baw068},\n  timestamp = {Thu, 13 Aug 2020 12:41:41
+0200},\n  biburl    = {https://dblp.org/rec/journals/biodb/LiSJSWLDMWL16.bib},\n  bibsource = {dblp computer science bibliography,
https://dblp.org}\n}\n',
    description='The BioCreative V Chemical Disease Relation (CDR) dataset is a large annotated text corpus of\nhuman annotations of
all chemicals, diseases and their interactions in 1,500 PubMed articles.\n',
    homepage='http://www.biocreative.org/tasks/biocreative-v/track-3-cdr/',
    license='Public Domain Mark 1.0'
)
```

### Loading datasets by config name

Each config helper provides a wrapper to the
[load_dataset](https://huggingface.co/docs/datasets/v2.2.1/en/package_reference/loading_methods#datasets.load_dataset)
function from Huggingface's [datasets](https://huggingface.co/docs/datasets/) package.
This wrapper will automatically populate the first two arguments of `load_dataset`,
 - `path`: path to the dataloader script
 - `name`: name of the dataset configuration

If you have a specific dataset and config in mind, you can,
 - fetch the helper from `conhelps` with the `for_config_name` method
 - load the dataset using the `load_dataset` wrapper

```python
bc5cdr_source = conhelps.for_config_name("bc5cdr_source").load_dataset()
bc5cdr_bigbio = conhelps.for_config_name("bc5cdr_bigbio_kb").load_dataset()
```

This wrapper function will pass through any other kwargs you may need to use.
For example `data_dir` for datasets that are not public,

```python
n2c2_2011_source = conhelps.for_config_name("n2c2_2011_source").load_dataset(data_dir="/path/to/n2c2_2011/data")
```


### Filtering by dataset properties

Some useful dataset discovery tools are available from the `BigBioConfigHeplers`

```python
# all dataset name
dataset_names = conhelps.available_dataset_names

# all dataset config names
ds_config_names = [helper.config.name for helper in conhelps]
```

You can use any attribute of a `BigBioConfigHelper` to filter the collection.
Here are some examples,

```python
# public bigbio configs that are not extra large
bb_public_helpers = conhelps.filtered(
    lambda x:
        x.is_bigbio_schema
	and not x.is_local
	and not x.is_large
)
```

```python
# source versions of all n2c2 datasets
n2c2_source_helpers = conhelps.filtered(
    lambda x:
        x.dataset_name.startswith("n2c2")
	and not x.is_bigbio_schema
)
```

```python
from bigbio.utils.constants import Tasks
nli_helpers = conhelps.filtered(
    lambda x: Tasks.TEXTUAL_ENTAILMENT in x.tasks
)
```

You can iterate over any instance of `BigBioConfigHelpers` and store the loaded datasets
in a container (e.g. a dictionary),

```python
bb_public_datasets = {
    helper.config.name: helper.load_dataset()
    for helper in bb_public_helpers
}
```

### Dataset metadata

The `BigBioConfigHelper` provides a `get_metadata` method that will calculate 
schema specific metadata for configs implementing a BigBIO schema. 
For example, 

```python
helper = conhelps.for_config_name('scitail_bigbio_te')
metadata = helper.get_metadata()
print(metadata)
{'train': BigBioTeMetadata(samples_count=23596, premise_char_count=2492695, hypothesis_char_count=1669028, label_counter={'neutral': 14994, 'entailment': 8602}), 'test': BigBioTeMetadata(samples_count=2126, premise_char_count=216196, hypothesis_char_count=153547, label_counter={'neutral': 1284, 'entailment': 842}), 'validation': BigBioTeMetadata(samples_count=1304, premise_char_count=138239, hypothesis_char_count=97320, label_counter={'entailment': 657, 'neutral': 647})}
```

## More Usage

At it's core, the bigbio package is a collection of
[dataloader scripts](https://huggingface.co/docs/datasets/dataset_script)
written to be used with the
[datasets](https://huggingface.co/docs/datasets/) package.


All of these scripts live in dataset specific directories in the `biogbio/biodatasets` folder.
If you cloned the repo, you will find the `bigbio` directory at the top level of the repo.
If you pip installed the package from github, the `bigbio` directory will be in
the `site-packages` directory of your python environment.


You can use them with the
[load_dataset](https://huggingface.co/docs/datasets/v2.2.1/en/package_reference/loading_methods#datasets.load_dataset)
function as you would any other local dataloader script if you like.
For example, if you cloned the repo

```python
from datasets import load_dataset
dsd = load_dataset("bigbio/biodatasets/anat_em/anat_em.py", name="anat_em_bigbio_kb")
```

In addition there is a convenience function that will provide the kwargs that should be used,

``` python
load_dataset_kwargs = conhelps.for_config_name("n2c2_2011_source").get_load_dataset_kwargs(data_dir="path/to_data")
print(load_dataset_kwargs)
{
    'path': '/home/user/repos/biomedical/bigbio/biodatasets/n2c2_2011/n2c2_2011.py',
    'name': 'n2c2_2011_source',
    'data_dir': 'path/to/data'
}
dsd = load_dataset(**load_dataset_kwargs)
```




## Benchmark Support

BigBIO includes support for almost all datasets included in other popular English biomedical benchmarks.

| Task Type |    Dataset    | BigBIO (Ours)|  BLUE | BLURB | In-BoXBART | Requires DUA |
|:---------:|:-------------:|:---------------------:|:-----:|:-----:|:----------:|:------------:|
|    NER    | BC5-chem      |          ✓         |  ✓ |  ✓ |    ✓    |         |
|    NER    | BC5-disease   |          ✓         |  ✓ |  ✓ |    ✓    |         |
|    NER    | NCBI-disease  |          ✓         |  |  ✓ |    ✓    |         |
|     RE    | ChemProt      |          ✓         |  ✓ |  ✓ |    ✓    |         |
|     RE    | DDI           |          ✓         |  ✓ |  ✓ |    ✓    |         |
|    STS   | BIOSSES       |          ✓         |  ✓ |  ✓ |       |         |
|     DC    | HoC           |          ✓         |  ✓ |  ✓ |    ✓    |         |
|     QA    | PubMedQA      |          ✓         |  |  ✓ |       |         |
|     QA    | BioASQ        |          ✓         |  |  ✓ |    ✓    |     ✓     |
|     TE    | MedSTS        |          TBD         |  ✓ |  |       |      ✓      |
|    NER   | ShARe/CLEFE   |          TBD         |  ✓ |  |       |     ✓       |
|    NER   | i2b2-2010     |          ✓         |  ✓ |  |       |     ✓     |
|    NLI   | MedNLI        |          ✓         |  ✓ |  |       |     ✓       |
|    NER    | BC2GM         |          ✓         |  |  ✓ |    ✓    |           |
|    NER    | JNLPBA        |          ✓         |  |  ✓ |    ✓    |           |
|    NER    | EBM PICO      |          ✓         |  |  ✓ |       |           |
|     RE    | GAD           |          ✓         |  |  ✓ |       |           |
|     SR    | Accelerometer |        Private         |  |  |    ✓    |           |
|     SR    | Acromegaly    | Private |  |  |    ✓    |           |
|    NER    | AnatEM        |          ✓         |  |  |    ✓    |           |
|     SR    | Cooking       |      Private        |  |  |    ✓    |           |
|    NER    | BC4CHEMD      |          ✓         |  |  |    ✓    |           |
|    NER    | BioNLP09      |          ✓         |  |  |    ✓    |           |
|    NER    | BioNLP11EPI   |          ✓         |  |  |    ✓    |           |
|    NER    | BioNLP11ID    |          ✓         |  |  |    ✓    |           |
|    NER    | BioNLP13CG    |          ✓         |  |  |    ✓    |           |
|    NER    | BioNLP13GE    |          ✓         |  |  |    ✓    |           |
|    NER    | BioNLP13PC    |          ✓         |  |  |    ✓    |           |
|     SR    | COVID         | Private |  |  |    ✓    |           |
|    NER    | CRAFT         |        TBD        |  |  |    ✓    |           |
|     DI    | DI-2006       |          ✓         |  |  |    ✓    |           |
|    NER    | Ex-PTM        |          ✓         |  |  |    ✓    |           |
|    POS    | Genia         |          ✓         |  |  |    ✓    |           |
|     SR    | HRT           | Private |  |  |    ✓    |           |
|    RFI    | RFHD-2014     |        TBD        |  |  |    ✓    |           |
|    NER    | Linnaeus      |          ✓         |  |  |    ✓    |           |
|     SA    | Medical Drugs |          ✓         |  |  |    ✓    |           |


## Citing
If you use BigBIO in your work, please cite

```
TBD
```


## Acknowledgements

BigBIO is a open source, community effort made possible through the efforts of many volunteers as part of BigScience and the [Biomedical Hackathon](HACKATHON.md).
