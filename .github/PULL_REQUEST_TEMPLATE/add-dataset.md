- **Name:** *name of the dataset*
- **Description:** *short description of the dataset (or link to social media or blog post)*
- **Paper:** *link to the dataset paper if available*
- **Data:** *link to the Github repository or current dataset location*
- **Motivation:** *what are some good reasons to have this dataset*

### Checkbox
**This is done using `datasets` format**
- [ ] Create the dataset script `/datasets/my_dataset/my_dataset.py` using the template
- [ ] Fill the `_DESCRIPTION` and `_CITATION` variables
- [ ] Implement `_infos()`, `_split_generators()` and `_generate_examples()` in dataloader script
- [ ] Make sure that the `BUILDER_CONFIGS` class attribute is filled with the different configurations of the dataset and that the `BUILDER_CONFIG_CLASS` is specified if there is a custom config class.
- [ ] Code works with `datasets`
- [ ] Passes all unit-tests for appropriate schema and keys
