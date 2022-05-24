"""
 Gather dataset statistics
Help make plots
"""
from collections import Counter
from collections import defaultdict, OrderedDict
import json

import bigbio
from bigbio.dataloader import BigBioConfigHelpers


MAX_COMMON = 50


def get_kb_meta(helper, split, ds):

    passages_count = 0
    passages_char_count = 0
    passages_type_counter = Counter()

    entities_count = 0
    entities_type_counter = Counter()
    entities_db_name_counter = Counter()
    entities_unique_db_ids = set()

    events_count = 0
    events_type_counter = Counter()
    events_arguments_count = 0
    events_arguments_role_counter = Counter()

    coreferences_count = 0

    relations_count = 0
    relations_type_counter = Counter()
    relations_db_name_counter = Counter()
    relations_unique_db_ids = set()

    for sample in ds:
        for passage in sample['passages']:
            passages_count += 1
            passages_char_count += len(passage["text"][0])
            passages_type_counter[passage["type"]] += 1

        for entity in sample['entities']:
            entities_count += 1
            entities_type_counter[entity['type']] += 1
            for norm in entity['normalized']:
                entities_db_name_counter[norm["db_name"]] += 1
                entities_unique_db_ids.add(norm["db_id"])

        for event in sample['events']:
            events_count += 1
            events_type_counter[event['type']] += 1
            for argument in event['arguments']:
                events_arguments_count += 1
                events_arguments_role_counter[argument["role"]] += 1

        for coreference in sample['coreferences']:
            coreferences_count += 1

        for relation in sample['relations']:
            relations_count += 1
            relations_type_counter[relation['type']] += 1
            for norm in relation['normalized']:
                relations_db_name_counter[norm["db_name"]] += 1
                relations_unique_db_ids.add(norm["db_id"])

    meta = {
        "split": split,
        "samples_count": ds.num_rows,
        "passages_count": passages_count,
        "passages_type_counts": dict(passages_type_counter.most_common(MAX_COMMON)),
        "passages_char_count": passages_char_count,
        "entities_count": entities_count,
        "entities_type_counts": dict(entities_type_counter.most_common(MAX_COMMON)),
        "entities_db_name_counts": dict(entities_db_name_counter.most_common(MAX_COMMON)),
        "entities_unique_db_id_counts": len(entities_unique_db_ids),
        "events_count": events_count,
        "events_arguments_count": events_arguments_count,
        "events_arguments_role_counts": dict(events_arguments_role_counter.most_common(MAX_COMMON)),
        "coreferences_count": coreferences_count,
        "relations_count": relations_count,
        "relations_type_counts": dict(relations_type_counter.most_common(MAX_COMMON)),
        "relations_db_name_counts": dict(relations_db_name_counter.most_common(MAX_COMMON)),
        "relations_unique_db_id_counts": len(relations_unique_db_ids),
    }

    return meta


def get_text_meta(helper, split, ds):

    text_char_count = 0
    labels_count = 0
    labels_counter = Counter()

    for sample in ds:
        text_char_count += len(sample["text"]) if sample["text"] is not None else 0
        for label in sample['labels']:
            labels_count += 1
            labels_counter[label] += 1

    meta = {
        "split": split,
        "samples_count": ds.num_rows,
        "text_char_count": text_char_count,
        "labels_count": labels_count,
        "labels_counts": dict(labels_counter.most_common(MAX_COMMON)),
    }

    return meta


def get_pairs_meta(helper, split, ds):

    text_1_char_count = 0
    text_2_char_count = 0
    label_counter = Counter()

    for sample in ds:
        text_1_char_count += len(sample["text_1"]) if sample["text_1"] is not None else 0
        text_2_char_count += len(sample["text_2"]) if sample["text_2"] is not None else 0
        label_counter[sample["label"]] += 1

    meta = {
        "split": split,
        "samples_count": ds.num_rows,
        "text_1_char_count": text_1_char_count,
        "text_2_char_count": text_2_char_count,
        "label_counts": dict(label_counter.most_common(MAX_COMMON)),
    }

    return meta


def get_qa_meta(helper, split, ds):

    question_char_count = 0
    context_char_count = 0
    answer_count = 0
    answer_char_count = 0
    type_counter = Counter()
    choices_counter = Counter()

    for sample in ds:
        question_char_count += len(sample["question"])
        context_char_count += len(sample["context"])
        type_counter[sample["type"]] += 1
        for choice in sample["choices"]:
            choices_counter[choice] += 1
        for answer in sample["answer"]:
            answer_count += 1
            answer_char_count += len(answer)

    meta = {
        "split": split,
        "samples_count": ds.num_rows,
        "question_char_count": question_char_count,
        "context_char_count": context_char_count,
        "answer_count": answer_count,
        "answer_char_count": answer_char_count,
        "type_counts": dict(type_counter.most_common(MAX_COMMON)),
        "choices_counts": dict(choices_counter.most_common(MAX_COMMON)),
    }

    return meta


def get_t2t_meta(helper, split, ds):

    text_1_char_count = 0
    text_2_char_count = 0
    text_1_name_counter = Counter()
    text_2_name_counter = Counter()

    for sample in ds:
        text_1_char_count += len(sample["text_1"]) if sample["text_1"] is not None else 0
        text_2_char_count += len(sample["text_2"]) if sample["text_2"] is not None else 0
        text_1_name_counter[sample["text_1_name"]] += 1
        text_2_name_counter[sample["text_2_name"]] += 1

    meta = {
        "split": split,
        "samples_count": ds.num_rows,
        "text_1_char_count": text_1_char_count,
        "text_2_char_count": text_2_char_count,
        "text_1_name_counts": dict(text_1_name_counter.most_common(MAX_COMMON)),
        "text_2_name_counts": dict(text_2_name_counter.most_common(MAX_COMMON)),
    }

    return meta


def get_te_meta(helper, split, ds):

    premise_char_count = 0
    hypothesis_char_count = 0
    label_counter = Counter()

    for sample in ds:
        premise_char_count += len(sample["premise"]) if sample["premise"] is not None else 0
        hypothesis_char_count += len(sample["hypothesis"]) if sample["hypothesis"] is not None else 0
        label_counter[sample["label"]] += 1

    meta = {
        "split": split,
        "samples_count": ds.num_rows,
        "premise_char_count": premise_char_count,
        "hypothesis_char_count": hypothesis_char_count,
        "label_counts": dict(label_counter.most_common(MAX_COMMON)),
    }

    return meta





# creating an instance of BigBioDataloader loads
# lots of metadata about the available datasets and configs
#==========================================================
conhelps = BigBioConfigHelpers()

conhelps = conhelps.filtered(lambda x: x.dataset_name != "pubtator_central")
conhelps = conhelps.filtered(lambda x: x.is_bigbio_schema)

print("loaded {} configs from {} datasets".format(
    len(conhelps),
    len(set([helper.dataset_name for helper in conhelps])),
))

public_conhelps = conhelps.filtered(lambda x: not x.is_local)
local_conhelps = conhelps.filtered(lambda x: x.is_local)


# when we actually read the datasets, we can examine more data
#==========================================================

conhelps_for_meta = public_conhelps
# conhelps_for_meta = local_conhelps


# gather configs by dataset
configs_by_ds = defaultdict(list)
for helper in conhelps_for_meta:
    configs_by_ds[helper.dataset_name].append(helper)


# now gather metadata
dataset_metas = {}
for dataset_name, helpers in configs_by_ds.items():
    print("dataset_name: ", dataset_name)

    config_metas = {}
    for helper in helpers:
        print("config name: ", helper.config.name)
        dsd = helper.load_dataset()

        split_metas = {}
        for split, ds in dsd.items():

            if helper.config.schema == "bigbio_kb":
                meta = get_kb_meta(helper, split, ds)

            elif helper.config.schema == "bigbio_text":
                meta = get_text_meta(helper, split, ds)

            elif helper.config.schema == "bigbio_t2t":
                meta = get_t2t_meta(helper, split, ds)

            elif helper.config.schema == "bigbio_pairs":
                meta = get_pairs_meta(helper, split, ds)

            elif helper.config.schema == "bigbio_qa":
                meta = get_qa_meta(helper, split, ds)

            elif helper.config.schema == "bigbio_te":
                meta = get_te_meta(helper, split, ds)

            else:
                raise ValueError()

            split_metas[split] = meta

        config_meta = {
            "config_name": helper.config.name,
            "bigbio_schema": helper.config.schema,
            "splits": split_metas,
            "splits_count": len(split_metas),
        }
        config_metas[helper.config.name] = config_meta


    dataset_meta = {
        "dataset_name": dataset_name,
        "is_local": False,
        "tasks": [el.name for el in helper.tasks],
        "languages": [el.name for el in helper.languages],
        "bigbio_version": helper.bigbio_version,
        "source_version": helper.source_version,
        "citation": helper.citation,
        "description": helper.description,
        "homepage": helper.homepage,
        "license": helper.license,
        "config_metas": config_metas,
        "configs_count": len(config_metas),
    }
    dataset_metas[dataset_name] = dataset_meta


with open("dataset_metadatas.json", "w") as fp:
    json.dump(dataset_metas, fp, indent=4)
