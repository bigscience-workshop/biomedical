# BigBIO: Biomedical Dataset Library

`BigBIO` (BigScience Biomedical) is an open library of biomedical dataloaders built using Huggingface's (🤗) [`datasets` library](https://huggingface.co/docs/datasets/) for data-centric machine learning. 

Our goals include:

- Lightweight, programmatic access to biomedical datasets at scale
- Promoting reproducibility in data processing
- Better documentation for dataset provenance, licensing, and other key attributes
- Easier generation of meta-datasets for natural language prompting, multi-task learning

Currently `BigBIO` provides support for:

- 126+ biomedical datasets
- 10+ languages
- 12 task categories
- Harmonized dataset schemas by task type
- Metadata on *licensing*, *coarse/fine-grained task types*, *domain*, and more!

## How to Use `BigBIO`

The preferred way to use these datasets is to access them from the [Official `BigBIO` Hub](https://huggingface.co/bigbio). 


Minimally, ensure you have the `datasets` library installed. Preferably, install the requirements as follows:

`pip install -r requirements.txt`.

<br>

You can access `BigBIO` datasets as follows:

```python
from datasets import load_dataset
data = load_dataset("bigbio/biosses")
```

In most cases, scripts load the original schema of the dataset by default. You can also access the `BigBIO` split that streamlines access to key information in datasets given a particular task. 

<br>

For example, the `biosses` dataset follows a `pairs` based schema, where text-based inputs (sentences, paragraphs) are assigned a "translated" pair. 

```python
from datasets import load_dataset
data = load_dataset("bigbio/biosses", name="biosses_bigbio_pairs")
```

Generally, you can load your datasets as follows:

```python
# Load original schema
data = load_dataset("bigbio/<your_dataset>")

# Load BigBIO schema
data = load_dataset("bigbio/<your_dataset_here>", name="<your_dataset>_bigbio_<schema_name>")
```

Check the datacards on the Hub to see what splits are available to you. You can find more information about [schemas](task_schemas.md) in [Documentation](##Documentation) below.

## Benchmark Support

`BigBIO` includes support for almost all datasets included in other popular English biomedical benchmarks.

| Task Type | Dataset       | [`BigBIO` (ours)](https://arxiv.org/abs/2206.15076) | [BLUE](https://arxiv.org/abs/1906.05474)  | [BLURB](https://microsoft.github.io/BLURB/) | [BoX](https://arxiv.org/abs/2204.07600) | DUA needed |
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| NER       | BC2GM         | ✓          |   | ✓  | ✓       |             |
| NER       | BC5-chem      | ✓          | ✓  | ✓  | ✓       |          |
| NER       | BC5-disease   | ✓          | ✓  | ✓  | ✓       |          |
| NER       | EBM PICO      | ✓          |   | ✓  |        |             |
| NER       | JNLPBA        | ✓          |   | ✓  | ✓       |             |
| NER       | NCBI-disease  | ✓          |   | ✓  | ✓       |          |
| RE        | ChemProt      | ✓          | ✓  | ✓  | ✓       |          |
| RE        | DDI           | ✓          | ✓  | ✓  | ✓       |          |
| RE        | GAD           | ✓          |   | ✓  |        |             |
| QA        | PubMedQA      | ✓          |   | ✓  |    ✓    |          |
| QA        | BioASQ        | ✓          |   | ✓  |  ✓       | ✓         |
| DC        | HoC           | ✓          | ✓  |   ✓  | ✓       |          |
| STS       | BIOSSES       | ✓          | ✓  |   ✓  |        |          |
| STS       | MedSTS        | *                | ✓  |   |        |   ✓          |
| NER       | n2c2 2010     | ✓          | ✓  |   |  ✓      | ✓         |
| NER       | ShARe/CLEF 2013   | *          | ✓  |   |        |   ✓          |
| NLI       | MedNLI        | ✓          | ✓  |   |        |    ✓         | 
| NER        | n2c2 deid 2006  | ✓          |   |   | ✓       |    ✓           |
| DC       | n2c2 RFHD 2014     | ✓       |   |   | ✓       |   ✓           |
| NER       | AnatEM        | ✓          |   |   | ✓       |             |
| NER       | BC4CHEMD      | ✓          |   |   | ✓       |             |
| NER       | BioNLP09      | ✓          |   |   | ✓       |             |
| NER       | BioNLP11EPI   | ✓          |   |   | ✓       |             |
| NER       | BioNLP11ID    | ✓          |   |   | ✓       |             |
| NER       | BioNLP13CG    | ✓          |   |   | ✓       |             |
| NER       | BioNLP13GE    | ✓          |   |   | ✓       |             |
| NER       | BioNLP13PC    | ✓          |   |   | ✓       |             |
| NER       | CRAFT         | *                |   |   | ✓       |             |
| NER       | Ex-PTM        | ✓          |   |   | ✓       |             |
| NER       | Linnaeus      | ✓          |   |   | ✓       |             |
| POS       | GENIA         | *                |   |   | ✓       |             |
| SA        | Medical Drugs | ✓          |   |   | ✓       |  |
| SR        | COVID         |          |   |   | private       |             |
| SR        | Cooking       |          |   |   | private      |             |
| SR        | HRT           |          |   |   | private      |             |
| SR        | Accelerometer |          |   |   | private       |             |
| SR        | Acromegaly    |          |   |   | private      |             |

\* denotes dataset implementation in-progress

## Documentation

- [Task Schema Overview](task_schemas.md) is an indepth explanation of `BigBIO` schemas implemented.

- [Streamlit Visualization Demo](https://github.com/bigscience-workshop/biomedical/tree/master/streamlit_demo)

- [BigBIO Data Cards](https://github.com/bigscience-workshop/biomedical/tree/master/figures/data_card) report on statistics around each dataset in the library.


## Tutorials

TBA - Links may not be applicable yet!

- Tutorials
  - [Materializing Meta-datasets](https://github.com/bigscience-workshop/biomedical/blob/master/notebooks/materializing_meta_datasets/materializing-meta-datasets.ipynb)   
  - [Prompt Engineering and Evaluation](https://github.com/bigscience-workshop/biomedical/tree/master/notebooks/promptengineering)  
  - [Prompt Engineering with BLOOM](notebooks/bloomprompting/bloompipeline.md)

## Contributing

`BigBIO` is an open source project - your involvement is warmly welcome! If you're excited to join us, we recommend the following steps:

- Looking for ideas? See our [Volunteer Project Board](https://github.com/orgs/bigscience-workshop/projects/6) to see what we may need help with.

- Have your own idea? Contact an admin in the form of an [issue](https://github.com/bigscience-workshop/biomedical/issues/new?assignees=&labels=&template=add-dataset.md&title=).

- Implement your idea following guidelines set by the [official contributing guide](CONTRIBUTING.md)

- Wait for admin approval; approval is iterative, but if accepted will belong to the main repository.

Currently, only admins will be merging all accepted changes to the Hub.

Feel free to join our [Discord](https://discord.com/invite/Cwf3nT3ajP)!

## Citing
If you use BigBIO in your work, please cite

```
@article{fries2022bigbio,
	title = {
		BigBIO: A Framework for Data-Centric Biomedical Natural Language
		Processing
	},
	author = {
		Fries, Jason Alan and Weber, Leon and Seelam, Natasha and Altay,
		Gabriel and Datta, Debajyoti and Garda, Samuele and Kang, Myungsun
		and Su, Ruisi and Kusa, Wojciech and Cahyawijaya, Samuel and others
	},
	journal = {arXiv preprint arXiv:2206.15076},
	year = 2022
}
```

## Acknowledgements

`BigBIO` is a open source, community effort made possible through the efforts of many volunteers as part of BigScience and the [Biomedical Hackathon](HACKATHON.md).
