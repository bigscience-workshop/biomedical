# makefile from the huggingface promptsource repository
# https://github.com/bigscience-workshop/promptsource/blob/main/Makefile
#
# run this Makefile on your dataset loader script,
# > make check_file=biodatasets/<dataset_name>/<dataset_name>.py

.PHONY: quality style

datasets_dir := biodatasets

# Check that source code meets quality standards (one file)

quality:
	black --check --line-length 119 --target-version py38 $(check_file)
	isort --check-only $(check_file)
	flake8 $(check_file) --max-line-length 119

# Format source code automatically (one file)

style:
	black --line-length 119 --target-version py38 $(check_file)
	isort $(check_file)

# Check that source code meets quality standards (all files)

quality_all:
	black --check --line-length 119 --target-version py38 $(datasets_dir)
	isort --check-only $(datasets_dir)
	flake8 $(datasets_dir) --max-line-length 119

# Format source code automatically (all files)

style_all:
	black --line-length 119 --target-version py38 $(datasets_dir)
	isort $(datasets_dir)
