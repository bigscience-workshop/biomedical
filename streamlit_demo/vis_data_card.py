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
import sys

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
    entry_id = entry['id']
    if schema == "bigbio_kb":
        for passage in entry["passages"]:
            result_key = passage["type"]
            for key in _TEXT_MAPS[schema]:
                text = passage[key][0]
                if not text:
                    print(f"WARNING: text key does not exist: entry {entry_id}")
                    result["token_length"] = 0
                    result["text_type"] = result_key
                    continue
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
                print(f"WARNING: text key does not exist, entry {entry_id}")
                result["token_length"] = 0
                result["text_type"] = key
                continue
            else:
                sents, ngrams = get_tuples_manual_sentences(text.lower(), N)
                toks = [tok for sent in sents for tok in sent]
                result["token_length"] = len(toks)
                result["text_type"] = key
                tups = ["_".join(tup) for tup in ngrams]
                counter.update(tups)
    return result, counter


def parse_token_length_and_n_gram(dataset, schema_type):
    hist_data = []
    n_gram_counters = []
    for split, data in dataset.items():
        n_gram_counter = Counter()
        for i, entry in enumerate(data):
            result, n_gram_counter = token_length_per_entry(
                entry, schema_type, n_gram_counter
            )
            result["split"] = split
            hist_data.append(result)
        n_gram_counters.append(n_gram_counter)

    return pd.DataFrame(hist_data), n_gram_counters


def resolve_splits(df_split):
    official_splits = set(df_split).intersection(set(SPLIT_COLOR_MAP.keys()))
    return official_splits


def draw_box(df, col_name, row, col, fig):
    splits = resolve_splits(df["split"].unique())
    for split in splits:
        split_count = df.loc[df["split"] == split, col_name].tolist()
        print(split)
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
    splits = resolve_splits(df["split"].unique())
    for split in splits:
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


def gen_latex(dataset_name, helper, splits, schemas, fig_path):
    if type(helper.description) is dict:
        # TODO hacky, change this to include all decsriptions
        descriptions = helper.description[list(helper.description.keys())[0]]
    else:
        descriptions = helper.description
    descriptions = descriptions.replace("\n", "").replace("\t", "")
    langs = [l.value for l in helper.languages]
    languages = " ".join(langs)
    if type(helper.license) is dict:
        license = helper.license.value.name
    else:
        license = helper.license.name
    tasks = [" ".join(t.name.lower().split("_")) for t in helper.tasks]
    tasks = ", ".join(tasks)
    schemas = " ".join([r"{\tt "] + list(schemas) + ["}"])  # TODO \tt
    splits = ", ".join(list(splits))
    data_name_display = " ".join(data_name.split("_"))
    latex_bod = r"\clearpage" + "\n" + r"\section*{" + fr"{data_name_display}" + " Data Card" + r"}" + "\n"
    latex_bod += (
        r"\begin{figure}[ht!]"
        + "\n"
        + r"\centering"
        + "\n"
        + r"\includegraphics[width=\linewidth]{"
    )
    latex_bod += f"{fig_path}" + r"}" + "\n"
    latex_bod += r"\caption{\label{fig:"
    latex_bod += fr"{data_name}" + r"}"
    latex_bod += (
        r"Token frequency distribution by split (top) and frequency of different kind of instances (bottom).}"
        + "\n"
    )
    latex_bod += r"\end{figure}" + "\n" + r"\textbf{Dataset Description} "
    latex_bod += (
        fr"{descriptions}"
        + "\n"
        + r"\textbf{Homepage:} "
        + f"{helper.homepage}"
        + "\n"
        + r"\textbf{URL:} "
        + f"{helper.homepage}"  # TODO change this later
        + "\n"
        + r"\textbf{Licensing:} "
        + f"{license}"
        + "\n"
        + r"\textbf{Languages:} "
        + f"{languages}"
        + "\n"
        + r"\textbf{Tasks:} "
        + f"{tasks}"
        + "\n"
        + r"\textbf{Schemas:} "
        + f"{schemas}"
        + "\n"
        + r"\textbf{Splits:} "
        + f"{splits}"
    )
    return latex_bod


def write_latex(latex_body, latex_name):
    text_file = open(f"tex/{latex_name}", "w")
    text_file.write(latex_body)
    text_file.close()


def draw_figure(data_name, data_config_name, schema_type):
    helper = conhelps.for_config_name(data_config_name)
    metadata_helper = helper.get_metadata()  # calls load_dataset for meta parsing
    rprint(metadata_helper)
    splits = metadata_helper.keys()
    # calls HF load_dataset _again_ for token parsing
    dataset = load_dataset(
        f"bigbio/biodatasets/{data_name}/{data_name}.py", name=data_config_name
    )
    # general token length
    tok_hist_data, ngram_counters = parse_token_length_and_n_gram(dataset, schema_type)
    rprint(helper)

    # general counter(s)
    # TODO generate the pdf and fix latex

    counters = parse_counters(metadata_helper)
    print(counters)
    rows = len(counters) // 3
    if len(counters) >= 3:
        # counters = counters[:3]
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
    if "token_length" in tok_hist_data.keys():
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
    fig.write_image(fig_path)
    dataset.cleanup_cache_files()

    return helper, splits, fig_path


if __name__ == "__main__":
    # load helpers
    # each entry in local metadata is the dataset name
    dc_local = load_helper(local="scripts/bigbio-public-metadatas-6-8.json")
    # each entry is the config
    conhelps = load_helper()
    dc = list()
    # TODO uncomment this
    # for conhelper in conhelps:
    #     # print(f"{conhelper.dataset_name}-{conhelper.config.subset_id}-{conhelper.config.schema}")
    #     dc.append(conhelper.dataset_name)

    # datacard per data, metadata chart per config
    # for data_name, meta in dc_local.items():
    #     config_metas = meta['config_metas']
    #     config_metas_keys = config_metas.keys()
    #     if len(config_metas_keys) > 1:
    #         print(f'dataset {data_name} has more than one config')
    #     schemas = set()
    #     for config_name, config in config_metas.items():
    #         bigbio_schema = config['bigbio_schema']
    #         helper, splits, fig_path = draw_figure(data_name, config_name, bigbio_schema)
    #         schemas.add(helper.bigbio_schema_caps)
    #         latex_bod = gen_latex(data_name, helper, splits, schemas, fig_path)
    #         latex_name = f"{data_name}_{config_name}.tex"
    #         write_latex(latex_bod, latex_name)
    #         print(latex_bod)

    # TODO try this code first, then use this for the whole loop
    data_name = sys.argv[1]
    meta = dc_local[data_name]
    config_metas = meta['config_metas']
    config_metas_keys = config_metas.keys()
    if len(config_metas_keys) >= 1:
        print(f'dataset {data_name} has more than one config')
    schemas = set()
    for config_name, config in config_metas.items():
        bigbio_schema = config['bigbio_schema']
        helper, splits, fig_path = draw_figure(data_name, config_name, bigbio_schema)
        schemas.add(helper.bigbio_schema_caps)
        latex_bod = gen_latex(data_name, helper, splits, schemas, fig_path)
        latex_name = f"{data_name}_{config_name}.tex"
        write_latex(latex_bod, latex_name)
        print(latex_bod)
