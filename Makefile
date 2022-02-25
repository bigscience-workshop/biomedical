# makefile from the huggingface promptsource repository
# https://github.com/bigscience-workshop/promptsource/blob/main/Makefile

.PHONY: quality style

check_dirs := examples templates

# Check that source code meets quality standards

quality:
	black --check --line-length 119 --target-version py38 $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 $(check_dirs) --max-line-length 119

# Format source code automatically

style:
	black --line-length 119 --target-version py38 $(check_dirs)
	isort $(check_dirs)