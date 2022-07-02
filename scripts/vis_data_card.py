# from matplotlib_venn import venn2, venn3
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from datasets import load_dataset
from plotly.subplots import make_subplots
from rich import print as rprint

from collections import Counter

from ngram import get_tuples_manual_sentences

from bigbio.dataloader import BigBioConfigHelpers

pio.kaleido.scope.mathjax = None


# vanilla tokenizer
def tokenizer(text, counter):
    if not text:
        return text, []
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


def load_helper(local=""):
    if local != "":
        with open(local, "r") as file:
            conhelps = json.load(file)
    else:
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
    "bigbio_pairs": ["text_1", "text_2"],
    "bigbio_t2t": ["text_1", "text_2"],
}

IBM_COLORS = [
    "#648fff",  # train
    "#dc267f",  # val
    "#ffb000",  # test
    "#fe6100",
    "#785ef0",
    "#000000",
    "#ffffff",
]

SPLIT_COLOR_MAP = {
    "train": "#648fff",
    "validation": "#dc267f",
    "test": "#ffb000",
}

N = 3


def token_length_per_entry(entry, schema, counter):
    result = {}
    if schema == "bigbio_kb":
        for passage in entry["passages"]:
            result_key = passage["type"]
            for key in _TEXT_MAPS[schema]:
                text = passage[key][0]
                sents, ngrams = get_tuples_manual_sentences(text.lower(), N)
                toks = [tok for sent in sents for tok in sent]
                tups = ["_".join(tup) for tup in ngrams]
                counter.update(tups)
                result["token_length"] = len(toks)
                result["text_type"] = result_key
    else:
        for key in _TEXT_MAPS[schema]:
            text = entry[key]
            if not text:
                rprint(entry)
                continue
            else:
                sents, ngrams = get_tuples_manual_sentences(text.lower(), N)
                toks = [tok for sent in sents for tok in sent]
                result["token_length"] = len(toks)
                result["text_type"] = key
                tups = ["_".join(tup) for tup in ngrams]
                counter.update(tups)
    return result, counter


def parse_token_length_and_n_gram(dataset, data_config):
    hist_data = []
    n_gram_counters = []
    for split, data in dataset.items():
        n_gram_counter = Counter()
        for i, entry in enumerate(data):
            result, n_gram_counter = token_length_per_entry(
                entry, data_config.schema, n_gram_counter
            )
            result["split"] = split
            hist_data.append(result)
            # print(result)
        n_gram_counters.append(n_gram_counter)

    return pd.DataFrame(hist_data), n_gram_counters


def center_title(fig):
    fig.update_layout(
        title={"y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"},
        font=dict(
            size=18,
        ),
    )
    return fig


def draw_box(df, col_name, row, col, fig):
    for split in df["split"].unique():
        split_count = df.loc[df["split"] == split, col_name].tolist()
        fig.add_trace(
            go.Box(
                x=split_count,
                name=split,
                marker_color=SPLIT_COLOR_MAP[split.split("_")[0]],
            ),
            row=row,
            col=col,
        )


def draw_bar(df, col_name, y_name, row, col, fig):
    for split in df["split"].unique():
        print(split)
        split_count = df.loc[df["split"] == split, col_name].tolist()
        y_list = df.loc[df["split"] == split, y_name].tolist()
        fig.add_trace(
            go.Bar(
                x=split_count,
                y=y_list,
                name=split,
                marker_color=SPLIT_COLOR_MAP[split.split("_")[0]],
                showlegend=False,
            ),
            row=row,
            col=col,
        )
    fig.update_traces(orientation="h")  # horizontal box plots


def parse_metrics(metadata):
    for k, m in metadata.items():
        mattrs = m.__dict__
        for m, attr in mattrs.items():
            if type(attr) == int and attr > 0:
                print(f"{k}-{m}: {attr}")


def parse_counters(metadata):
    metadata = metadata[
        list(metadata.keys())[0]
    ]  # using the training counter to fetch the names
    counters = []
    for k, v in metadata.__dict__.items():
        if "counter" in k and len(v) > 0:
            counters.append(k)
    return counters


# generate the df for histogram
def parse_label_counter(metadata, counter_type):
    hist_data = []
    for split, m in metadata.items():
        metadata_counter = getattr(m, counter_type)
        for k, v in metadata_counter.items():
            row = {}
            row["labels"] = k
            row[counter_type] = v
            row["split"] = split
            hist_data.append(row)
    return pd.DataFrame(hist_data)


def draw_figure(data_name, data_config_name):
    helper = conhelps.for_config_name(data_config_name)
    metadata_helper = helper.get_metadata()
    rprint(metadata_helper)

    parse_metrics(metadata_helper)

    # load HF dataset
    data_idx = data_config_names.index(data_config_name)
    data_config = data_configs[data_idx]
    dataset = load_dataset(
        f"bigbio/biodatasets/{data_name}/{data_name}.py", name=data_config_name
    )
    # general token length
    tok_hist_data, ngram_counters = parse_token_length_and_n_gram(dataset, data_config)

    # general counter(s)
    counters = parse_counters(metadata_helper)
    rows = len(counters) // 3
    if len(counters) >= 3:
        counters = counters[:3]
        cols = 3
        specs = [[{"colspan": 3}, None, None]] + [[{}, {}, {}]] * (rows + 1)
    elif len(counters) == 1:
        specs = [[{}], [{}]]
        cols = 1
    elif len(counters) == 2:
        specs = [[{"colspan": 2}, None]] + [[{}, {}]] * (rows + 1)
        cols = 2
    counters.sort()

    counter_titles = ["Label Counts by Type: " + ct.split("_")[0] for ct in counters]
    titles = ("token length",) + tuple(counter_titles)
    # Make figure with subplots
    fig = make_subplots(
        rows=rows + 2,
        cols=cols,
        subplot_titles=titles,
        specs=specs,
        vertical_spacing=0.10,
        horizontal_spacing=0.10,
    )
    # draw token distribution
    draw_box(tok_hist_data, "token_length", row=1, col=1, fig=fig)
    for i, ct in enumerate(counters):
        row = i // 3 + 2
        col = i % 3 + 1
        label_df = parse_label_counter(metadata_helper, ct)
        label_min = int(label_df[ct].min())
        # filter_value = int((label_max - label_min) * 0.01 + label_min)
        label_df = label_df[label_df[ct] >= label_min]
        print(label_df.head(5))

        # draw bar chart for counter
        draw_bar(label_df, ct, "labels", row=row, col=col, fig=fig)

    # add annotation
    descriptions = helper.description.replace("\n", "").replace("\t", "")
    langs = [l.value for l in helper.languages]
    languages = " ".join(langs)
    license = helper.license.value.name
    tasks = [" ".join(t.name.lower().split("_")) for t in helper.tasks]
    tasks = ", ".join(tasks)

    fig.update_annotations(font_size=12)
    fig.update_layout(
        margin=dict(l=25, r=25, t=25, b=25, pad=2),
        # showlegend=False,
        # title_text=data_name,
        height=600,
        width=1000,
    )

    # fig.show()

    fig_name = f"{data_name}_{data_config_name}.pdf"
    fig_path = f"figures/data_card/{fig_name}"
    data_name_display = " ".join(data_name.split("_"))
    latex_bod = r"\textbf{" + fr"{data_name_display}" + r"}" + "\n"
    latex_bod += (
        r"\begin{figure}[ht!]"
        + "\n"
        + r"\centering"
        + "\n"
        + r"\includegraphics[width=\linewidth]{"
    )
    latex_bod += f"{fig_path}" + r"}" + "\n"
    latex_bod += r"""\caption{\label{fig:"""
    latex_bod += fr"{data_name}" + r"}"
    latex_bod += (
        r"Token frequency distribution by split (top) and Frequency of different kind of instances (bottom).}"
        + "\n"
    )
    latex_bod += r"\end{figure}" + "\n" + r"\paragraph{Dataset Description}"
    latex_bod += (
        fr"{descriptions}"
        + "\n"
        + r"\paragraph{Licensing} "
        + f"{license}"
        + "\n"
        + r"\paragraph{Languages} "
        + f"{languages}"
        + "\n"
        + r"\paragraph{Tasks} "
        + f"{tasks}"
    )

    fig.write_image(fig_path)

    latex_name = f"{data_name}_{data_config_name}.tex"

    text_file = open(f"tex/{latex_name}", "w")
    n = text_file.write(latex_bod)
    text_file.close()
    print(latex_bod)
    dataset.cleanup_cache_files()


if __name__ == "__main__":
    # load helpers
    conhelps_local = load_helper(local="scripts/bigbio-public-metadatas-6-8.json")
    conhelps = load_helper()
    configs = list()

    for conhelper in conhelps:
        configs.append(conhelper.dataset_name)
    names = ["mqp", "paramed", "mediqa_qa", "scitail"]
    # TODO: parse all public dataset
    for data_name in names:
        # data_name = configs['data_name']
        # data_info = conhelps_local[data_name]
        # setup data configs
        data_helpers = conhelps.for_dataset(data_name)
        data_configs = [d.config for d in data_helpers]
        data_config_names = [d.config.name for d in data_helpers]
        for data_config_name in data_config_names:
            draw_figure(data_name, data_config_name)
