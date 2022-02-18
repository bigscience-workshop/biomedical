from datasets import load_dataset


ds_source = load_dataset(
    "n2c2_2011_coref",
    name="source",
    data_dir="/home/galtay/data/big_science_biomedical/n2c2_2011_coref",
)

ds_bigbio = load_dataset(
    "n2c2_2011_coref",
    name="bigbio",
    data_dir="/home/galtay/data/big_science_biomedical/n2c2_2011_coref",
)


num_entities = 0
for sample in ds_bigbio["train"]:

    top_id = sample["id"]
    doc_id = sample["document_id"]

    passage_text = sample['passages'][0]['text'][0]
    entity_lookup = {ent["id"]: ent for ent in sample["entities"]}

    # check all coref entity ids are in entity lookup
    for coref in sample['coreferences']:
        for entity_id in coref['entity_ids']:
            assert entity_id in entity_lookup
            entity = entity_lookup[entity_id]
            ii, ff = entity['offsets'][0][0], entity['offsets'][0][1]
            ptext = passage_text[ii:ff]

    for entity in sample['entities']:
        ii, ff = entity['offsets'][0][0], entity['offsets'][0][1]
        ptext = passage_text[ii:ff]
        ctext = entity['text'][0]
        assert ptext.lower() == ctext
        num_entities += 1

print(num_entities)
