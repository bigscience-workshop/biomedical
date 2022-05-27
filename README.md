# BigBIO: Biomedical Datasets

BigBIO (BigScience Biomedical) is an open library of biomedical dataloaders built using Huggingface's (🤗) [`datasets` library](https://huggingface.co/docs/datasets/) for data-centric. Our goals include

- Lightweight, programmatic access to biomedical datasets at scale
- Promote reproducibility in data processing
- Better documentation for dataset provenance, licensing, and other key attributes
- Easier generation of meta-datasets (e.g., prompting, masssive MTL)

Currently BigBIO provides support for

- 127 biomedical datasets
- X languages
- Harmonized dataset schemas by task type
- Metadata on *licensing*, *coarse/fine-grained task types*, *domain*, and more!   


### Documentation

- Tutorials: *Coming Soon!*
- [Volunteer Project Board](https://github.com/orgs/bigscience-workshop/projects/6): Implement or suggest new datasets
- [Contributor Guide](CONTRIBUTING.md)
- [Task Schema Overview](task_schemas.md)


## Installation 

Using pip

```
pip install -U git+https://github.com/bigscience-workshop/biomedical.git
```
or git clone this repo and the install

```
git clone git@github.com:bigscience-workshop/biomedical.git
cd biomedical
pip install -e .
```

Using conda

```
conda env create --name bigbio --file=conda.yml
```

## Usage

Load a specific public dataset (here the [BioCreative V Chemical-Disease Relation task](https://biocreative.bioinformatics.udel.edu/tasks/biocreative-v/track-3-cdr/))

```
from datasets import load_dataset

# load the original source schema
ds_source = load_dataset('bc5cdr', name='bc5cdr_source')

# load the harmonized biobio schema
ds_bb = load_dataset('bc5cdr', name='bc5cdr_bigbio_kb')
```

Load a specific private (local) dataset 

```
from datasets import load_dataset

# local datasets require a path
ds_source = load_dataset('bioasq_task_b', data_dir='', name='bc5cdr_source')
```

Get all available dataset configuration names

```
from bigbio.dataloader import BigBioConfigHelpers

ds_names = [x.config.name for x in BigBioConfigHelpers()]
```


Load a filtered subset of all public datasets. Warning, this will take some time to download!

```
from bigbio.dataloader import BigBioConfigHelpers
conhelps = BigBioConfigHelpers()

# filter on dataset attributes
bb_public_helpers = conhelps.filtered(
	lambda x: x.is_bigbio_schema and not x.is_local and not x.is_large
)
for helper in bb_public_helpers:
    dsd = helper.load_dataset()
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

BigBIO is a open source, community effort made possible through the efforts of many volunteers as part of BigScience and the [Biomedical Hackathon](https://github.com/bigscience-workshop/data_tooling/wiki/datasets-hackathon).
