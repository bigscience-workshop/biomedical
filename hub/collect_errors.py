import json

from datasets import load_dataset
from datasets import get_dataset_config_names
from huggingface_hub import get_repo_discussions

from hubtools import list_datasets


ds_infos = list_datasets()

prs = {}
errors = {}
works = {}

for ds_info in ds_infos:
    print(ds_info)
    ds_configs = get_dataset_config_names(ds_info.id)
    print(ds_configs)
    discussions = list(get_repo_discussions(repo_id=ds_info.id, repo_type="dataset"))
    prs[ds_info.id] = discussions
    print()
    for config in ds_configs:

        if ds_info.id == "bigscience-biomedical/pubtator_central" and config != "pubtator_central_sample_source":
            continue

        try:
            dsd = load_dataset(ds_info.id, name=config)
            works[(ds_info.id, config)] = "works"
        except BaseException as oops:
            errors[(ds_info.id, config)] = oops


json.dump({"|".join(k): str(v) for k,v in errors.items()}, open("errors.json", "w"), indent=4)
json.dump({"|".join(k): str(v) for k,v in works.items()}, open("works.json", "w"), indent=4)
