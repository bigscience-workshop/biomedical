# Dataloader
Contains a collection of biomedical datasets that are accessible through a unified interface. When possible, a dataset is automatically downloaded and cached when first used.

## Available datasets

### Information Extraction
| Name                                                                                                                                                                                                                                                                                                | Entity Annotations                                                       | Relation Annotations                                                                                                                           | Event Annotations | 
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|
| jnlpba                                                                                                                                                                                                                                                                                              | Protein, DNA, RNA, Cell Type, Cell Line                                  | -                                                                                                                                              | -                 |
| cellfinder                                                                                                                                                                                                                                                                                          | Anatomy, Cell Compartment., Cell Line, Cell Type, Gene/Protein, Species  | -                                                                                                                                              | -                 |
| linneaus                                                                                                                                                                                                                                                                                            | Species                                                                  | -                                                                                                                                              | -                 |
| chemprot                                                                                                                                                                                                                                                                                            | Chemical, Protein                                                        | Part of, Regulator, Upregulator and activator, Downregulator and inhibitor, Agonist, Antagonist, Modulator, Cofactor, Substrate and product of | -                 |
| ddi                                                                                                                                                                                                                                                                                                 | Drug, Brand, Group, Drug_n (active substance not approved for human use) | Effect, Mechanism, Advice, int (unspecified interaction)                                                                                       | -                 | -
| bc5cdr                                                                                                                                                                                                                                                                                              | Chemical, Disease                                                        | CID (Chemical-induced disease)                                                                                                                 | -                 | -
| hunflair_chemical_chebi hunflair_chemical_cdr hunflair_chemical_cemp hunflair_chemical_scai                                                                                                                                                                                                         | Chemical                                                                 | -                                                                                                                                              | -                 | -
| hunflair_disease_cdr hunflair_disease_pdr hunflair_disease_ncbi hunflair_disease_scai hunflair_disease_mirna hunflair_disease_variome                                                                                                                                                               | Disease                                                                  | -                                                                                                                                              | -                 |
| hunflair_gene_variome hunflair_gene_fsu hunflair_gene_gpro hunflair_gene_deca hunflair_gene_iepa hunflair_gene_bc2gm hunflair_gene_bio_infer hunflair_gene_cellfinder hunflair_gene_chebi hunflair_gene_craftv4 hunflair_gene_jnlpba hunflair_gene_loctext hunflair_gene_mirna hunflair_gene_osiris | Gene/Protein                                                             | -                                                                                                                                              | -                 |
| hunflair_species_cellfinder hunflair_species_s800 hunflair_species_chebi hunflair_species_mirna hunflair_species_loctext hunflair_species_craftv4 hunflair_species_linneaus hunflair_species_variome                                                                                                | Species                                                                  | -                                                                                                                                              | -                 |
## Loading data sets

### Information Extraction
Load the dataset named `jnlpba` and cache it `data/datasets`:

```python
from dataloader.dataset import BioDataset

dataset = BioDataset("jnlpba", data_root="data/datasets")
```

Access splits with `dataset.data.train`, `dataset.data.dev` and `dataset.data.test`, each of which is a
[pybrat](https://github.com/Yevgnen/pybrat) `Example`. Note, that `dataset` will throw an
`AttributeError` if the split does not exist for the specific dataset.

The annotations in each example can be accessed as follows

```python
example = dataset.data.train[0]

# Entities
entity = example.entities[0]
entity.mention == example.text[entity.start:entity.end] # start and end are character offsets 
entity.type

# Relations
relation = example.relations[0]
relation.arg1: Entity
relation.arg2: Entity
relation.type: str

# Events
event = example.events[0]
event.type: str
event.trigger: Entity

argument = event.arguments[0]
argument.rule: str # typo in pybrat, should be 'role'
argument.entity: Union[Entity, Event] # misnomer, can be either Entity or Event
```

## Adding a new data set

### Information Extraction
The only hard requirement for a newly added dataset is that it has to expose the interface
defined in [Loading data sets](#Loading-data-sets).
Under the hood, this usually means that it takes care of three steps:
1. Download dataset
2. Transform dataset to brat and cache in `data_root`
3. Load dataset with `pybrat` to populate `self.data.train`, `self.data.dev` and `self.data.test` if these splits exist

Finally, add the dataset to the [datasets list](#Available-datasets)

See `dataloader.utils.dataloaders.ChemProt` for a prototypical example.