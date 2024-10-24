import sys
from bigbio.hf_maps import BATCH_MAPPERS_TEXT_FROM_SCHEMA
from bigbio.dataloader import BigBioConfigHelpers
from datasets import load_dataset
from nomic import atlas
import pandas as pd


S_MAX = 3200


def load_conhelps():
    conhelps = BigBioConfigHelpers()
    conhelps = conhelps.filtered(lambda x: not x.is_large)
    conhelps = conhelps.filtered(lambda x: x.is_bigbio_schema)
    conhelps = conhelps.filtered(lambda x: not x.is_local)
    return conhelps


def load_data(conhelp):
    try:
        dsd = conhelp.load_dataset()
    except:
        return None
    dsd = dsd.map(
        BATCH_MAPPERS_TEXT_FROM_SCHEMA[conhelp.bigbio_schema_caps.lower()],
        batched=True,
    )
    return dsd


conhelps = load_conhelps()
df_all = pd.DataFrame()
for ii, conhelp in enumerate(conhelps):

    dsd = load_data(conhelp)
    print(conhelp.config)
    if dsd is None:
        print("skipping, could not load")
        continue

    if conhelp.languages != ["English"]:
        continue
    if 'train' not in dsd.keys():
        print(f"SKIPPING {conhelp.display_name}")
        continue

    dsd = dsd.remove_columns([
        col for col in dsd.column_names['train']
        if col not in ["id", "text"]
    ])

    dfs = []
    for split, ds in dsd.items():
        df1 = ds.to_pandas()
        df1["split"] = split
        df1["dataset"] = conhelp.display_name
        df1['schema'] = conhelp.config.schema
        df1['config_name'] = conhelp.config.name
        dfs.append(df1)

    df = pd.concat(dfs)
    df = df.reset_index(drop=True)
    df = df.rename(columns={"id": "sample_id"})
    if df.shape[0] > S_MAX:
        df = df.sample(n=S_MAX)

    df_all = pd.concat([df_all, df])
#    if ii >= 50:
#        break




project = atlas.map_text(
    data=df_all.to_dict(orient="records"),
    indexed_field='text',
    name='bigbio',
    colorable_fields=['dataset', 'split', 'schema', 'config_name'],
    description='BigBIO',
    reset_project_if_exists=True,
)

map = project.get_map('bigbio')
print(map)


sys.exit(0)


# this is how to add more
new_df_all = pd.DataFrame()
# add stuff to data frame
with project.wait_for_project_lock():
    project.add_text(data=new_df_all.to_dict(orient="records"))
    project.rebuild_maps()
