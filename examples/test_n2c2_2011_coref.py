from datasets import load_dataset


dso = load_dataset(
    "n2c2_2011_coref.py",
    name="original",
    data_dir="/home/galtay/data/big_science_biomedical/n2c2_2011_coref",
)

dsbb = load_dataset(
    "n2c2_2011_coref.py",
    name="bigbio-kb",
    data_dir="/home/galtay/data/big_science_biomedical/n2c2_2011_coref",
)

dsbbl = load_dataset(
    "n2c2_2011_coref.py",
    name="bigbio-kb-list",
    data_dir="/home/galtay/data/big_science_biomedical/n2c2_2011_coref",
)


num_entities = 0
for sample in dsbb["train"]:

    top_id = sample["id"]
    doc_id = sample["document_id"]

    passage_id = sample["passages"]["id"][0]
    passage_type = sample["passages"]["type"][0]
    passage_text = sample["passages"]["text"][0]
    passage_offset = sample["passages"]["offsets"][0][0]

    entity_ids = sample["entities"]["id"]
    entity_offsets = [el[0] for el in sample["entities"]["offsets"]]
    entity_texts = sample["entities"]["text"]

    coref_ids = sample["coreferences"]["id"]
    coref_entity_ids = sample["coreferences"]["entity_ids"]

    entity_lookup = {ent_id: ii for ii, ent_id in enumerate(entity_ids)}

    # check all coref entity ids are in entity lookup
    for ii_coref, coref_id in enumerate(coref_ids):
        for entity_id in coref_entity_ids[ii_coref]:
            assert entity_id in entity_lookup
            ii_entity = entity_lookup[entity_id]
            ii, ff = entity_offsets[ii_entity][0], entity_offsets[ii_entity][1]
            ptext = passage_text[ii:ff]
            print(ptext)
        print("*" * 30)

    for ii_entity, entity_id in enumerate(entity_ids):
        ii, ff = entity_offsets[ii_entity][0], entity_offsets[ii_entity][1]
        ptext = passage_text[ii:ff]
        ctext = entity_texts[ii_entity]
        assert ptext.lower() == ctext
        num_entities += 1


print(num_entities)
