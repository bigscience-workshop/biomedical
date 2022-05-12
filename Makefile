# makefile from the huggingface promptsource repository
# https://github.com/bigscience-workshop/promptsource/blob/main/Makefile
#
# run this Makefile on your dataset loader script,
# > make check_file=biodatasets/<dataset_name>/<dataset_name>.py

.PHONY: quality

datasets_dir := bigbio/biodatasets
examples_dir := examples

# Format source code automatically (one file)

quality:
	black --line-length 119 --target-version py38 $(check_file)
	isort $(check_file)
	flake8 $(check_file) --max-line-length 119

# Format source code automatically (all files)

quality_all:
	black --check --line-length 119 --target-version py38 $(datasets_dir)
	isort --check-only $(datasets_dir)
	flake8 $(datasets_dir) --max-line-length 119
	black --check --line-length 119 --target-version py38 $(examples_dir)
	isort --check-only $(examples_dir)
	flake8 $(examples_dir) --max-line-length 119
