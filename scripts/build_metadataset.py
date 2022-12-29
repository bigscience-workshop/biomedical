import re

import datasets
from datasets import concatenate_datasets
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm
import requests

from bigbio.hub.hubtools import get_dataset_infos
from bigbio.hub.hubtools import list_datasets
from bigbio.hf_maps import BATCH_MAPPERS_TEXT_FROM_SCHEMA


SCHEMAS = ["kb", "text", "pairs", "qa", "t2t", "te"]
DATASETS_SERVER_API_URL = "https://datasets-server.huggingface.co/splits?dataset="
STREAMING = False


def query_splits_from_ds_server(dataset_id, url=DATASETS_SERVER_API_URL):
    response = requests.request("GET", f"{url}{dataset_id}")
    return response.json()


def fetch_public_dataset_info():
    dataset_infos_all = list_datasets(full=True)
    dataset_infos = {
        dsi.id: dsi for dsi in dataset_infos_all if dsi.cardData["bigbio_public"]
    }
    return dataset_infos


def get_ds_metas(dataset_infos, split_name):

    logger.info(f"get_ds_metas with split_name={split_name}")

    ds_metas = {}
    for dsid, dsi in tqdm(dataset_infos.items()):
        ds_name = dsid.split("/")[1]
        config_pattern = "{}_bigbio_({})".format(ds_name, "|".join(SCHEMAS))
        api_res = query_splits_from_ds_server(dsid)

        if not "splits" in api_res:
            logger.warning(f"skipping {ds_name} because of bad API response.")
            continue

        good_split = None
        for split in api_res["splits"]:

            if split["split"] != split_name:
                continue

            match = re.match(config_pattern, split["config"])
            if match is None:
                continue

            schema = match.groups()[0]
            good_split = split
            logger.info(f"found good split {good_split}")
            break

        if good_split is not None:
            ds_metas[dsid] = {}
            ds_metas[dsid]["info"] = dsi
            ds_metas[dsid]["splits"] = api_res["splits"]
            ds_metas[dsid]["good_split"] = good_split
            ds_metas[dsid]["schema"] = schema
        else:
            continue

    return ds_metas


def load_datasets(ds_metas):

    o_ds = {}
    t_ds = {}
    for dsid, ds_meta in ds_metas.items():
        logger.info("loading {} from {}".format(dsid, ds_meta["good_split"]))
        if dsid in ("bigbio/tmvar_v2", "bigbio/bioscope", "bigbio/meqsum"):
            continue

        ds_name = dsid.split("/")[1]
        ds = load_dataset(
            dsid,
            name=ds_meta["good_split"]["config"],
            split=ds_meta["good_split"]["split"],
            streaming=STREAMING,
        )
        o_ds[ds_name] = ds
        t_ds[ds_name] = ds.map(
            BATCH_MAPPERS_TEXT_FROM_SCHEMA[ds_meta["schema"]],
            remove_columns=ds.features.keys(),
            batched=True,
        )

    return o_ds, t_ds


def is_ok_str(example):
    is_str = isinstance(example["text"], str)
    if not is_str:
        return False
    is_informative = len(example["text"].strip()) > 0
    return is_str & is_informative


dataset_infos = fetch_public_dataset_info()
ds_metas = {}
ds_o = {}
ds_t = {}
ds_concat_t = {}
for split_name in ["train", "validation", "test"]:
    ds_metas[split_name] = get_ds_metas(dataset_infos, split_name)
    ds_o[split_name], ds_t[split_name] = load_datasets(ds_metas[split_name])

    logger.info(
        "top 10 datasets by samples:\n {}".format(
            sorted([(v.num_rows, k) for k, v in ds_t[split_name].items()])[-10:]
        )
    )

    ds = concatenate_datasets(list(ds_t[split_name].values()))
    logger.info(f"ds initially has {ds.num_rows} rows")
    ds = ds.filter(is_ok_str)
    logger.info(f"ds after text filter has {ds.num_rows} rows")
    ds_concat_t[split_name] = ds


dsd = datasets.DatasetDict(ds_concat_t)
for split_name, ds in dsd.items():
    ds.save_to_disk(f"bigbio-public-text-concat-{split_name}")
