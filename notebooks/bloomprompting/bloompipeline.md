# Running with the Bloom Pipeline
**DATE: 2022.09.25**

Creating a pipeline to run prompt evals of bigbio via BLOOM

## Step 1: Make a conda environment and install necessary packages.

Create a base package and activate it.

```
conda create --name bloom python=3.8.3 -y
conda activate bloom
```

Then, I advise you to make a new directory to store all the various libraries etc. We will be installing a few packages from source.

```
mkdir bloom_pipeline
cd bloom_pipeline
```

Check to see that your `python` and `pip` installation point to the environment's installation. Then, install python packages

You will need 2 packages to install. The first is `lm-evaluation-harness` or LmEval.

Install as follows (as per instructions provided):
```
git clone https://github.com/bigscience-workshop/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e ".[dev]"
cd ..
```

The second is the `eval-hackathon` branch of PromptSource; lm-eval supports this branch of PromptSource specifically (note this install command is borrowed from the recommendations of the `lm-evaluation-harness` README:)

```
git clone --single-branch --branch eval-hackathon https://github.com/bigscience-workshop/promptsource
cd promptsource
pip install -e .
```

I also went ahead and installed any extra dependencies in case (these are probably installed anyway for the most part):

```
pip install torch transformers scikit-learn pandas numpy scipy datasets tensorboard ipython
```

## Step 1.5: Create prompts for your dataset

You will need to use [promptsource](https://github.com/OpenBioLink/promptsource/tree/main/promptsource) to make custom prompts. Since we are just borrowing tasks from scitail, I will not cover this. Please visit their documentation, or alternatively learn how to make prompts from our earlier [tutorial](https://github.com/bigscience-workshop/biomedical/tree/main/notebooks/promptengineering).

## Step 2: Make a task suited to your dataset

### <span style="color:red">**TLDR**. Move the contents to `lm_eval/tasks` into your installation of `lm_evaluation_harness`.</span>

<br>
We now need to make a task via the PromptSourceTask template. To do this, we make a `Task` file and place it in `lm-evaluation-harness/lm_eval/tasks`.

We follow the [recipe](https://github.com/bigscience-workshop/lm-evaluation-harness/blob/master/templates/new_prompt_source_task.py) provided as follows:

```python
"""
2022.09.25
N. Seelam

This path tests the `bigbio` version of SciTail. We generate prompts

"""
from lm_eval.api.task import PromptSourceTask

_CITATION = """"""

class BBSciTail(PromptSourceTask):

    DATASET_PATH = "bigscience-biomedical/scitail"
    DATASET_NAME = "scitail_bigbio_te"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            return self.dataset["train"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

```
Copy the above and move it to the tasks folder as `bigbioscitail.py`.

Some notes:
    - We delete the `max_generation_length` since we will not be running a generation task
    - The dataset I tried has train/val/test docs, hence I kept all of the `has_<x>_docs` functions `True`
    - We do not use advanced features of the following functions: `construct_requests, process_results, aggregation, higher_is_better`

In order for this task to be recognized, we need to include it into the task registry. To do this, we need to include it in the `__init__.py` file of the tasks. 

Open this file: `lm-evaluation-harness/lm_eval/tasks/__init__.py`.

Add your task in the task registry as such

```python
from . import bigbioscitail
TASK_REGISTRY = {
    ...
    "your_dataset_name": yourdataset.Class_Corresponding_To_Schema
}
```

Generically this is:
```python
from . import <your_dataset>  # Place this in the beginning import

# Within TASK_REGISTRY, add the following command
TASK_REGISTRY = {
    ...
    "bigbioscitail": bigbioscitail.BBSciTail
}
```

## Step 3: Make your prompts

### <span style="color:red">**TLDR**. Move the template file in `promptsource/templates.yaml` into your installation of promptsource. Place this file in the folder: `promptsource/promptsource/templates/bigscience-biomedical/scitail/scitail-bigbio-te`</span>

<br>
I will not cover how to make a prompt a priori since we will be borrowing a prompt from [here](https://github.com/OpenBioLink/promptsource/tree/main/promptsource/templates). Specifically, I will modify the `scitail` prompt.

The prompt is below:

```yaml
dataset: bigscience-biomedical/scitail
subset: scitail_bigbio_te
templates:
  ea58d4dc-4a46-4419-8312-4ba5961c0260: !Template
    answer_choices: yes ||| no
    id: ea58d4dc-4a46-4419-8312-4ba5961c0260
    jinja: 'Given that {{premise}} Does it follow that {{hypothesis}}
      {{ answer_choices | join('' or '') }}?
      |||{% if label == "entailment" %}
      {{answer_choices[0]}}
      {% else %}
      {{answer_choices[1]}}
      {% endif %}'
    metadata: !TemplateMetadata
      choices_in_prompt: true
      metrics:
      - Accuracy
      original_task: true
    name: Yes/No Entailment Framing
    reference: ''
```

Since we are importing from the bigbio Hub version of datasets, we need to make a matching path. Enter your promptsource installation (note, should be local as we git cloned this) and go to the templates folder. This should be found here: `promptsource/promptsource/templates`.

BigBio is hosted on the hub, meaning if we load datasets, we will use the command `bigscience-biomedical/scitail`. The templates file needs to have this structure, additionally there is another folder to indicate the specific data split. We will be using the `scitail_bigbio_te` split.

This means we will need to make a directory as such and copy the yaml file into it:

```
mkdir -p promptsource/promptsource/templates/bigscience-biomedical/scitail/scitail_bigbio_te
```

## Step 4: Run the model

Run your model with the appropriate BLOOM checkpoint:

python main.py \
    --model_api_name 'hf-causal' \
    --model_args use_accelerate=True,pretrained='bigscience/bigscience-small-testing' \
    --task_name bigbioscitail

You can use any of the specific arguments (ex: `template_names`) as necessary.
