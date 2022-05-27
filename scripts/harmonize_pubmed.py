"""
Harmonize pubmed datasets
"""
from collections import Counter
from collections import defaultdict, OrderedDict
import json
import random

import pandas as pd

import bigbio
from bigbio.dataloader import BigBioConfigHelpers


def pubnorm_just_pmid(document_id):
    return "PMID", document_id


def pubnorm_anat_em(document_id):
    pieces = document_id.split("-")
    if pieces[0] == "PMID":
        source = "PMID"
    elif pieces[0] == "PMC":
        source = "PMCID"
    return source, pieces[1]


def pubnorm_an_em(document_id):
    pieces = document_id.split("-")
    if pieces[0] == "PMID":
        source = "PMID"
    elif pieces[0] == "PMC":
        source = "PMCID"
    return source, pieces[1]


def pubnorm_bioasq_task_b(document_id):
    return "PMID", document_id.split("/")[-1]


def pubnorm_bionlp_st_2011_epi(document_id):
    return "PMID", document_id.split("-")[1]


def pubnorm_bionlp_st_2011_ge(document_id):
    pieces = document_id.split("-")
    if pieces[0] == "PMID":
        source = "PMID"
    elif pieces[0] == "PMC":
        source = "PMCID"
    return source, pieces[1]


def pubnorm_bionlp_st_2011_id(document_id):
    return "PMCID", document_id.split("-")[0].replace("PMC", "")


def pubnorm_bionlp_st_2011_rel(document_id):
    return "PMID", document_id.split("-")[1]


def pubnorm_bionlp_st_2013_cg(document_id):
    return "PMID", document_id.split("-")[1]


def pubnorm_bionlp_st_2013_ge(document_id):
    pieces = document_id.split("-")
    if pieces[0] == "PMID":
        source = "PMID"
    elif pieces[0] == "PMC":
        source = "PMCID"
    return source, pieces[1]


def pubnorm_bionlp_st_2013_gro(document_id):
    return "PMID", document_id.split("-")[1]


def pubnorm_bionlp_st_2013_pc(document_id):
    return "PMID", document_id.split("-")[1]


def pubnorm_bionlp_st_2019_bb(document_id):
    pieces = document_id.split("-")
    if pieces[2] == "F":
        return "PMID", pieces[-2]
    else:
        return "PMID", pieces[-1]


def pubnorm_cellfinder(document_id):
    return "PMID", document_id.split("_")[0]


def pubnorm_genia_relation_corpus(document_id):
    return "PMID", document_id.split("-")[1]


def pubnorm_linnaeus(document_id):
    return "PMID", document_id.replace("pmcA", "")


def pubnorm_lll(document_id):
    return "PMID", document_id.split("-")[0]


def pubnorm_mantra_gsc_medline(document_id):
    return "PMID", document_id.split("_")[1].split(".")[0][1:]


def pubnorm_mlee(document_id):
    return "PMID", document_id.split("-")[1]


def pubnorm_pico_extraction(document_id):
    return "PMID", document_id.split(":")[0]


def pubnorm_verspoor(document_id):
    return "PMID", document_id.split("-")[0]


_DATASET_DOC_ID_TO_PUBMED = {
    "anat_em": pubnorm_anat_em,
    "an_em": pubnorm_an_em,
    "bc5cdr": pubnorm_just_pmid,
    "bc7_litcovid": pubnorm_just_pmid,
    "bioasq_task_b": pubnorm_bioasq_task_b,
    "bioasq_task_c_2017": pubnorm_just_pmid,
    "bionlp_shared_task_2009": pubnorm_just_pmid,
    "bionlp_st_2011_epi": pubnorm_bionlp_st_2011_epi,
    "bionlp_st_2011_ge": pubnorm_bionlp_st_2011_ge,
    "bionlp_st_2011_id": pubnorm_bionlp_st_2011_id,
    "bionlp_st_2011_rel": pubnorm_bionlp_st_2011_rel,
    "bionlp_st_2013_cg": pubnorm_bionlp_st_2013_cg,
    "bionlp_st_2013_ge": pubnorm_bionlp_st_2013_ge,
    "bionlp_st_2013_gro": pubnorm_bionlp_st_2013_gro,
    "bionlp_st_2013_pc": pubnorm_bionlp_st_2013_pc,
    "bionlp_st_2019_bb": pubnorm_bionlp_st_2019_bb,
    "biored": pubnorm_just_pmid,
    "cellfinder": pubnorm_cellfinder,
    "chebi_nactem": pubnorm_just_pmid,
    "chemdner": pubnorm_just_pmid,
    "chemprot": pubnorm_just_pmid,
    "ebm_pico": pubnorm_just_pmid,
    "genia_relation_corpus": pubnorm_genia_relation_corpus,
    "gnormplus": pubnorm_just_pmid,
    "hallmarks_of_cancer": pubnorm_just_pmid,
    "hprd50": pubnorm_just_pmid,
    "iepa": pubnorm_just_pmid,
    "linnaeus": pubnorm_linnaeus,
    "lll": pubnorm_lll,
    "medmentions": pubnorm_just_pmid,
    "mlee": pubnorm_mlee,
    "mutation_finder": pubnorm_just_pmid,
    "ncbi_disease": pubnorm_just_pmid,
    "nlmchem": pubnorm_just_pmid,
    "nlm_gene": pubnorm_just_pmid,
    "osiris": pubnorm_just_pmid,
    "pdr": pubnorm_just_pmid,
    "pico_extraction": pubnorm_pico_extraction,
    "pubtator_central": pubnorm_just_pmid,
    "scai_chemical": pubnorm_just_pmid,
    "scai_disease": pubnorm_just_pmid,
    "seth_corpus": pubnorm_just_pmid,
    "thomas2011": pubnorm_just_pmid,
    "tmvar_v1": pubnorm_just_pmid,
    "tmvar_v2": pubnorm_just_pmid,
    "tmvar_v3": pubnorm_just_pmid,
    "verspoor": pubnorm_verspoor,
}


_CONFIG_DOC_ID_TO_PUBMED = {
    "quaero_medline_bigbio_kb": pubnorm_just_pmid,
    'mantra_gsc_es_medline_bigbio_kb': pubnorm_mantra_gsc_medline,
    'mantra_gsc_fr_medline_bigbio_kb': pubnorm_mantra_gsc_medline,
    'mantra_gsc_de_medline_bigbio_kb': pubnorm_mantra_gsc_medline,
    'mantra_gsc_nl_medline_bigbio_kb': pubnorm_mantra_gsc_medline,
    'mantra_gsc_en_medline_bigbio_kb': pubnorm_mantra_gsc_medline,
}


# creating an instance of BigBioConfigHelpers loads
# lots of metadata about the available datasets and configs
# ==========================================================
conhelps = BigBioConfigHelpers()
conhelps = conhelps.filtered(lambda x: x.dataset_name != "pubtator_central")
conhelps = conhelps.filtered(lambda x: x.is_bigbio_schema)
conhelps = conhelps.filtered(
    lambda x: (
        x.dataset_name in _DATASET_DOC_ID_TO_PUBMED
        or x.config.name in _CONFIG_DOC_ID_TO_PUBMED
    )
)

print(
    "loaded {} configs from {} datasets".format(
        len(conhelps),
        len(set([helper.dataset_name for helper in conhelps])),
    )
)

public_conhelps = conhelps.filtered(lambda x: not x.is_local)
local_conhelps = conhelps.filtered(lambda x: x.is_local)


# lets map out all the pubmed IDs
# ==========================================================

conhelps_for_pubmed = public_conhelps
# conhelps_for_meta = local_conhelps


# gather configs by dataset
configs_by_ds = defaultdict(list)
for helper in conhelps_for_pubmed:
    configs_by_ds[helper.dataset_name].append(helper)


# now gather metadata
rows = []
dss = {}
for dataset_name, helpers in configs_by_ds.items():
    print("dataset_name: ", dataset_name)

    config_metas = {}
    for helper in helpers:
        print("config name: ", helper.config.name)

        doc_to_pid_func = _DATASET_DOC_ID_TO_PUBMED.get(dataset_name)
        if doc_to_pid_func is None:
            doc_to_pid_func = _CONFIG_DOC_ID_TO_PUBMED.get(helper.config.name)

        dsd = helper.load_dataset()

        split_metas = {}
        for split_index, (split, ds) in enumerate(dsd.items()):

            if split_index == 0:
                dss[helper.config.name] = ds
                for nn in range(10):
                    ii = random.randint(0, ds.num_rows - 1)
                    sample = ds[ii]
                    document_id = sample["document_id"]
                    source, pid = doc_to_pid_func(document_id)
                    print(
                        "   {: <10} {: >45} {: >12} {: >12}".format(
                            ii, document_id, source, pid
                        )
                    )
                print()

            for sample_index, sample in enumerate(ds):
                document_id = sample["document_id"]
                source, pid = doc_to_pid_func(document_id)
                if source == "PMID":
                    pmid = int(pid)
                    pmcid = pd.NA
                elif source == "PMCID":
                    pmid = pd.NA
                    pmcid = f"PMC{pid}"

                row = (
                    dataset_name,
                    helper.config.name,
                    split,
                    sample_index,
                    document_id,
                    pmid,
                    pmcid,
                )
                rows.append(row)


df = pd.DataFrame(
    rows,
    columns=[
        "dataset_name",
        "config_name",
        "split",
        "sample_index",
        "docid",
        "pmid",
        "pmcid",
    ],
)


# read local copy of pubmed xwalk
# https://www.ncbi.nlm.nih.gov/pmc/tools/id-converter-api/
# https://ftp.ncbi.nlm.nih.gov/pub/pmc/PMC-ids.csv.gz
#=======================================================================
df_pmc_ids = pd.read_csv("PMC-ids.csv")
df_pmc_ids["PMID"] = df_pmc_ids["PMID"].astype("Int64")
df["pmid"] = df["pmid"].astype("Int64")


# split df into rows that have PMID and rows that have PMCID
#=======================================================================
df_pmid = df[~df["pmid"].isnull()]
df_pmcid = df[~df["pmcid"].isnull()]


# join each to pubmed x-walk
#=======================================================================
df_all_pmid = pd.merge(df_pmid, df_pmc_ids, left_on="pmid", right_on="PMID", how="left")
df_all_pmcid = pd.merge(df_pmcid, df_pmc_ids, left_on="pmcid", right_on="PMCID", how="left")

df_all_pmid["pmcid"] = df_all_pmid["PMCID"]
df_all_pmcid["pmid"] = df_all_pmcid["PMID"]

df_all = pd.concat([df_all_pmid, df_all_pmcid])


df_grpd = (
    df_all[["dataset_name", "pmid"]]
    .drop_duplicates()
    .groupby("pmid")
    .agg(frozenset)
    .reset_index()
)
df_grpd["num_datasets"] = df_grpd["dataset_name"].apply(len)
df_most_common = (
    df_grpd[df_grpd["num_datasets"] > 1].groupby("dataset_name").size().sort_values()
)


df_all.to_csv("bigbio-public-pubmed-meta.csv", index=False)
