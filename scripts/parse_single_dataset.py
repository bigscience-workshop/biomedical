import sys
from datasets import load_dataset
import streamlit as st
import time
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
from rich import print as rprint

import plotly.express as px
import pandas as pd

from bigbio.dataloader import BigBioConfigHelpers
from collections import defaultdict, OrderedDict, Counter


# vanilla tokenizer
def tokenizer(text):
    text = text.strip()
    text = text.replace("\t", "")
    text = text.replace("\n", "")
    # split
    text_list = text.split(" ")
    return text, text_list


def norm(lengths):
    mu = np.mean(lengths)
    sigma = np.std(lengths)
    return mu, sigma


def load_helper():
    conhelps = BigBioConfigHelpers()
    conhelps = conhelps.filtered(lambda x: x.dataset_name != "pubtator_central")
    conhelps = conhelps.filtered(lambda x: x.is_bigbio_schema)
    conhelps = conhelps.filtered(lambda x: not x.is_local)
    rprint(
        "loaded {} configs from {} datasets".format(
            len(conhelps),
            len(set([helper.dataset_name for helper in conhelps])),
        )
    )
    return conhelps


_TEXT_MAPS = {
    "bigbio_kb": ["text"],
    "bigbio_text": ["text"],
    "bigbio_qa": ["question", "context"],
    "bigbio_te": ["premise", "hypothesis"],
    "bigbio_tp": ["text_1", "text_2"],
    "bigbio_t2t": ["text_1", "text_2"],
}

IBM_COLORS = [
    "#648fff",
    "#785ef0",
    "#dc267f",
    "#fe6100",
    "#ffb000",
    "#000000",
    "#ffffff",
]


def token_length_per_entry(entry, schema):
    result = {}
    if schema == "bigbio_kb":
        for passage in entry["passages"]:
            result_key = passage['type']
            for key in _TEXT_MAPS[schema]:
                text = passage[key][0]
                _, toks = tokenizer(text)
                result[result_key] = len(toks)

    else:
        for key in _TEXT_MAPS[schema]:
            text = entry[key]
            _, toks = tokenizer(text)
            result[key] = len(toks)
    return result


def count_label(entry, schema, counter):
    if schema == "bigbio_kb":
        for e in entry["entities"]:
            label = e['text'][0].lower()
            counter[label] += 1
        for ev in entry['events']:
            label = ev['trigger']['text'][0].lower()
            counter[label] += 1
        for re in entry["relations"]:
            label = re['type'].lower()
            counter[label] += 1
    elif schema == "bigbio_qa":
        label = entry["answer"][0].lower()
        counter[label] += 1
    elif schema == "bigbio_te" or schema == "bigbio_tp":
        counter[entry["label"].lower()] += 1
    elif schema == "bigbio_text":
        for label in entry["labels"]:
            counter[label.lower()] += 1
    return counter


def parse_token_length(dataset, data_config, st=None):
    hist_data = []
    rprint(data_config)
    for split, data in dataset.items():
        my_bar = st.progress(0)
        total = len(data)
        for i, entry in enumerate(data):
            my_bar.progress(int(i / total * 100))
            result = token_length_per_entry(entry, data_config.schema)
            result['total_token_length'] = sum([v for k, v in result.items()])
            result["split"] = split
            hist_data.append(result)
        my_bar.empty()
    st.write('token lengths complete!')
    return pd.DataFrame(hist_data)


def parse_labels(dataset, data_config, st=None):
    split_counter = {}
    for split, data in dataset.items():
        my_bar = st.progress(0)
        counter = Counter()
        total = len(data)
        for i, entry in enumerate(data):
            my_bar.progress(int(i / total * 100))
            counter = count_label(entry, data_config.schema, counter)
        split_counter[split] = counter
        my_bar.empty()
    st.write('labels complete!')
    return split_counter


def draw_token_dist(dataset, config, st=None):
    hist_data = parse_token_length(dataset, config, st.sidebar)

    fig = px.histogram(
        hist_data,
        x="total_token_length",
        color="split",
        color_discrete_sequence=IBM_COLORS,
        marginal="box",  # or violin, rug
        barmode="group",
        hover_data=hist_data.columns,
    )
    st.plotly_chart(fig, use_container_width=True)


def draw_label_pie(dataset, config, st):
    label_counter = parse_labels(dataset, config, st.sidebar)

    cols = st.columns(len(label_counter))

    for (s, lc), col in zip(label_counter.items(), cols):
        most_x = 7
        most_popular = lc.most_common()[:most_x]
        # rest = lc.most_common()[most_x:]
        # rest_sum = sum([v for k, v in rest])
        # most_popular.append(('rest', rest_sum))
        pie_data = dict(most_popular)
        col.subheader(f'Top {most_x} popular labels in {s}')

        pie_fig = go.Figure(data=[go.Pie(labels=list(pie_data.keys()), values=list(pie_data.values()), hole=.3, marker_colors=IBM_COLORS)])
        col.plotly_chart(pie_fig, use_container_width=True)


def parse_metrics(metadata_helper, st=None):
    metadata = helper.get_metadata()
    print(metadata['train'].__dict__)
    for k, m in metadata.items():
        mattrs = m.__dict__
        for m, attr in mattrs.items():
            if type(attr) == int and attr > 0:
                st.metric(label=f'{k}-{m}', value=attr)  #, delta=-0.5, delta_color="inverse")


if __name__ == "__main__":
    # load helpers
    conhelps = load_helper()
    configs_set = set()

    for conhelper in conhelps:
        configs_set.add(conhelper.dataset_name)

    # setup page, sidebar, columns
    st.set_page_config(layout="wide")
    data_name = st.sidebar.selectbox("dataset", configs_set)
    st.sidebar.write("you selected:", data_name)
    st.header(f"Dataset stats for {data_name}")
    # data_name = "chemdner"

    # setup data configs
    data_helpers = conhelps.for_dataset(data_name)
    data_configs = [d.config for d in data_helpers]
    data_config_names = [d.config.name for d in data_helpers]
    data_config_name = st.sidebar.selectbox("dataset", set(data_config_names))
    # data_config_name = "chemdner_bigbio_kb"

    # test without streamlit
    # helper = conhelps.for_config_name(data_config_name)
    # helper.get_metadata()
    # rprint(helper.get_metadata())

    # data_idx = data_config_names.index(data_config_name)
    # dataset = load_dataset(
    #     f"bigbio/biodatasets/{data_name}/{data_name}.py", name=data_config_name)
    # hist_data = parse_token_length(dataset, data_configs[data_idx])
    # rprint(hist_data.head())
    # label_counter = parse_labels(dataset, data_configs[data_idx])
    # print(label_counter)

    if st.sidebar.button("fetch"):
        helper = conhelps.for_config_name(data_config_name)
        parse_metrics(helper, st.sidebar)

        # load HF dataset
        data_idx = data_config_names.index(data_config_name)
        data_config = data_configs[data_idx]
        dataset = load_dataset(
            f"bigbio/biodatasets/{data_name}/{data_name}.py", name=data_config_name)

        # draw token distribution
        draw_token_dist(dataset, data_config, st)
        # draw label distribution
        draw_label_pie(dataset, data_config, st)

    st.sidebar.button("Re-run")
