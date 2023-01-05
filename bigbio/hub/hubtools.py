"""
https://huggingface.co/docs/huggingface_hub/how-to-manage
"""

import os
from huggingface_hub import create_repo
from huggingface_hub import HfApi


HF_ORG = "bigbio"
HF_DATASETS_URL_BASE = "https://huggingface.co/datasets"


def list_datasets(full=False):
    """List datasets

    https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.list_datasets
    """
    api = HfApi()
    bb_hub_datasets = api.list_datasets(author=HF_ORG, full=full)
    return bb_hub_datasets


def get_dataset_infos(dataset_ids):
    """Get dataset info

    https://huggingface.co/docs/huggingface_hub/main/en/package_reference/hf_api#huggingface_hub.HfApi.dataset_info
    """
    api = HfApi()
    ds_infos = [
        api.dataset_info(dataset_id)
        for dataset_id in dataset_ids
    ]
    return ds_infos


def create_repository(dataset_name, private=True):
    """Create a new dataset repository in the hub.

    https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.create_repo
    """
    repo_id = os.path.join(HF_ORG, dataset_name)
    create_repo(repo_id, repo_type="dataset", private=private)


def upload_bigbiohub(dataset_names=None, dryrun=True):
    """Upload bigbiohub.py to one or more hub dataset repositories.

    https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.upload_file
    """

    if dataset_names is None:
        repo_ids = [dataset.id for dataset in list_datasets()]
    else:
        repo_ids = [
            os.path.join(HF_ORG, dataset_name)
            for dataset_name in dataset_names
        ]

    local = "bigbiohub.py"

    print(f"going to upload {local} to the folling repos {repo_ids}")
    if dryrun:
        print("this is a dryrun. to actually upload, run again with dryrun=False")
        return

    api = HfApi()
    for repo_id in repo_ids:
        print(f"uploading {local} to {repo_id}")
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=local,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"upload {local} to hub from bigbio repo",
            commit_description=f"upload {local} to hub from bigbio repo",
        )


def upload_file(dataset_name, local_path, repo_path, dryrun=True):
    """Upload a file to hub dataset repository.

    https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.upload_file
    """

    repo_id = os.path.join(HF_ORG, dataset_name)

    print(f"going to upload {local_path} to {repo_id} as {repo_path}")
    if dryrun:
        print("this is a dryrun. to actually upload, run again with dryrun=False")
        return

    api = HfApi()
    print(f"uploading {local_path} to {repo_id}")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"upload {local_path} to hub from bigbio repo",
        commit_description=f"upload {local_path} to hub from bigbio repo",
    )



if __name__ == "__main__":

    bb_datasets = list_datasets()

#    dataset_name = "testing_private"
#    result = create_repository(dataset_name, private=True)
