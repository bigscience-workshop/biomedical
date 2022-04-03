Please name your PR after the issue it closes. You can use the following line: "Closes #ISSUE-NUMBER" where you replace the ISSUE-NUMBER with the one corresponding to your dataset.

If the following information is NOT present in the issue, please populate:

- **Name:** *name of the dataset*
- **Description:** *short description of the dataset (or link to social media or blog post)*
- **Paper:** *link to the dataset paper if available*
- **Data:** *link to the online home of the dataset*

### Checkbox

- [ ] Create the dataloader script `biodatasets/my_dataset/my_dataset.py` (please use only lowercase and underscore for dataset naming).
- [ ] Provide values for the `_CITATION`, `_DATASETNAME`, `_DESCRIPTION`, `_HOMEPAGE`, `_LICENSE`, `_URLs`, `_SUPPORTED_TASKS`, `_SOURCE_VERSION`, and `_BIGBIO_VERSION` variables.
- [ ] Implement `_info()`, `_split_generators()` and `_generate_examples()` in dataloader script.
- [ ] Make sure that the `BUILDER_CONFIGS` class attribute is a list with at least one `BigBioConfig` for the source schema and one for a bigbio schema.
- [ ] Confirm dataloader script works with `datasets.load_dataset` function
- [ ] Confirm that your dataloader script passes the test suite run with `python -m tests.test_bigbio biodatasets/my_dataset/my_dataset.py`
