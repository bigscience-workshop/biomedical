"""
Pull all files from HF Hub repos to local

A bit hacky
Clones all the repos using git and then removes the hidden git folders.

You can do this with the HF Hub Client
https://huggingface.co/docs/huggingface_hub/package_reference/file_download
but it will put the files in the cache.
"""

import os
import subprocess
from bigbio.hub.hubtools import list_datasets, HF_DATASETS_URL_BASE, HF_ORG


ds_infos = list_datasets()
for ds_info in ds_infos:

    ds_name = ds_info.id.replace(HF_ORG+"/", "")
    local_path = os.path.join("hub_repos", ds_name)
    print(local_path)

    repo_url = os.path.join(HF_DATASETS_URL_BASE, ds_info.id)
    print(repo_url)

    subprocess.run(["rm", "-rf", local_path])
    subprocess.run(["git", "clone", repo_url, local_path])
    subprocess.run(["rm", "-rf", os.path.join(local_path, ".git")])
    subprocess.run(["rm", "-f", os.path.join(local_path, ".gitattributes")])
