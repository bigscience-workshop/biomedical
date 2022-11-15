import os
from huggingface_hub import create_repo
from huggingface_hub import HfApi


HF_ORG = "bigscience-biomedical"


def list_datasets():
    """List datasets

    https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.list_datasets
    """
    api = HfApi()
    bb_hub_datasets = api.list_datasets(author=HF_ORG)
    return bb_hub_datasets


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



if __name__ == "__main__":

    bb_datasets = list_datasets()

#    dataset_name = "testing_private"
#    result = create_repository(dataset_name, private=True)

    upload_bigbiohub()
