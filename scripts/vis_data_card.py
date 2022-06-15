import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datasets import load_dataset
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
from rich import print as rprint
from matplotlib import pyplot as plt
from matplotlib_venn import venn2, venn3
import json
import plotly.express as px
import pandas as pd

# import plotly.plotly as py
import plotly.tools as tls
import plotly.io as pio

pio.kaleido.scope.mathjax = None

from bigbio.dataloader import BigBioConfigHelpers
from collections import Counter

from ngram import get_tuples_manual_sentences


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
                marker_color=SPLIT_COLOR_MAP[split],
            ),
            row=row,
            col=col,
        )


def draw_bar(df, col_name, y_name, row, col, fig):
    for split in df["split"].unique():
        split_count = df.loc[df["split"] == split, col_name].tolist()
        y_list = df.loc[df["split"] == split, y_name].tolist()
        fig.add_trace(
            go.Bar(
                x=split_count,
                y=y_list,
                name=split,
                marker_color=SPLIT_COLOR_MAP[split],
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
    metadata = metadata["train"]  # using the training counter to fetch the names
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
    rows = len(counters) + 1
    titles = ('token length',) + tuple(' '.join(ct.split('_')[:-1]) for ct in counters)
    # Make figure with subplots
    fig = make_subplots(
        rows=rows,
        cols=1,
        subplot_titles=titles,
        specs=[[{"type": "box"}]] + [[{"type": "bar"}]] * len(counters),
    )
    # draw token distribution
    draw_box(tok_hist_data, "token_length", row=1, col=1, fig=fig)
    for i, ct in enumerate(counters):
        label_df = parse_label_counter(metadata_helper, ct)
        label_max = int(label_df[ct].max() - 1)
        label_min = int(label_df[ct].min())
        filter_value = int((label_max - label_min) * 0.1 + label_min)
        label_df = label_df[label_df[ct] >= filter_value]
        # draw bar chart for counter
        draw_bar(label_df, ct, "labels", row=i+2, col=1, fig=fig)

    # add annotation
    # counter = str(counter_type)
    # counter = f'<b>Label Count by</b> ' + ' '.join(counter_type.split('_')[:-1])
    descriptions = helper.description.replace('\n', '').replace('\t', '')
    # descriptions = f'<b>Data description </b> ' + descriptions
    langs = [l.value for l in helper.languages]
    languages = ' '.join(langs)
    # languages = '<b>Supported Languages</b> ' + languages
    license = helper.license.value.name
    # license = '<b>Licensed by </b>' + license
    tasks = [' '.join(t.name.lower().split('_')) for t in helper.tasks]
    tasks = ', '.join(tasks)
    # tasks = '<b>Supported Tasks</b> ' + tasks
    # events = [descriptions, counter, license, languages, tasks]
    # anns = []
    # for i, e in enumerate(events):
    #     ann = go.layout.Annotation(
    #         # bordercolor='black',  # Remove this to hide border
    #         align='left',  # Align text to the left
    #         yanchor='bottom', # Align text box's top edge
    #         text=e,  # Set text with '<br>' strings as newlines
    #         showarrow=False, # Hide arrow head
    #         # width=1000, # Wrap text at around 800 pixels
    #         xref='paper',  # Place relative to figure, not axes
    #         yref='paper',
    #         # font={'family': 'Courier'},  # Use monospace font to keep nice indentation
    #         x=0,  # Place on left edge
    #         y=-0.5-(i/10)  # Place a little more than half way down
    #     )
    #     anns.append(ann)

    fig.update_annotations(font_size=12)
    fig.update_layout(
        showlegend=False,
        # title_text=data_name,
        height=800,
        width=800,
    )   

    fig.show()

    fig_name = f"{data_name}_{data_config_name}.pdf"
    fig_path = f"figures/data_card/{fig_name}"
    data_name_display = ' '.join(data_name.split('_'))
    latex_bod = r"\textbf{" + fr'{data_name_display}' + r'}' + '\n'
    latex_bod += r"\begin{figure}[ht!]" + "\n" + r"\centering" + "\n" + r"\includegraphics[width=\linewidth]{"
    latex_bod += f'{fig_path}' + r'}' + '\n'
    latex_bod += r'''\caption{\label{fig:'''
    latex_bod += fr'{data_name}' + r'}' 
    latex_bod += r"Token frequency distribution by split (top) and Frequency of different kind of instances (bottom).}" + "\n"
    latex_bod += r"\end{figure}" + "\n" + r"\paragraph{Dataset Description}"
    latex_bod += fr'{descriptions}' + '\n' + r'\paragraph{Licensing} ' + f'{license}' + '\n' + r'\paragraph{Languages} ' + f'{languages}' + '\n' + r'\paragraph{Tasks} ' + f'{tasks}'


    # fig.write_image(fig_path)

    # latex_name = f"{data_name}_{data_config_name}.tex"

    # text_file = open(f"tex/{latex_name}", "w")
    # n = text_file.write(latex_bod)
    # text_file.close()
    print(latex_bod)


if __name__ == "__main__":
    # load helpers
    conhelps_local = load_helper(local="scripts/bigbio-public-metadatas-6-8.json")
    conhelps = load_helper()
    configs = list()

    for conhelper in conhelps:
        configs.append(conhelper.dataset_name)

    # for data_name in configs:
    data_name = configs[0]  # TODO CHANGE THIS
    data_info = conhelps_local[data_name]
    # setup data configs
    data_helpers = conhelps.for_dataset(data_name)
    data_configs = [d.config for d in data_helpers]
    data_config_names = [d.config.name for d in data_helpers]
    # data_config_name = data_config_names[0]  # TODO CHANGE THIS
    for data_config_name in data_config_names:
        draw_figure(data_name, data_config_name)

