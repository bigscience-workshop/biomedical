import sys
from datasets import load_dataset
import streamlit as st
import time
import numpy as np
import plotly.figure_factory as ff


# vanilla tokenizer
def tokenizer(text):
    text = text.strip()
    text = text.replace('\t', '')
    text = text.replace('\n', '')
    # split
    text_list = text.split(' ')
    return text, text_list


def norm(lengths):
    mu = np.mean(lengths)
    sigma = np.std(lengths)
    return mu, sigma


def load_data(arg):
    dataset = load_dataset(f'bigbio/biodatasets/{arg}/{arg}.py')
    hist_data = []
    for split, data in dataset.items():
        split_data = []
        for i, entry in enumerate(data):
            _, tok_list = tokenizer(entry['passages'][-1]['text'])
            split_data.append(len(tok_list))
        split_data = np.array(split_data)
        hist_data.append(split_data)
    return hist_data, list(dataset.keys())


if __name__ == "__main__":
    arg = sys.argv[1]
    print(arg)
    data = load_dataset(f'bigbio/biodatasets/{arg}/{arg}.py')
    print(data['train'][0])
    print(len(data['train']))

    # line plot with progress bar
    # progress_bar = st.sidebar.progress(0)
    # status_text = st.sidebar.empty()
    # last_rows = np.random.randn(1, 1)
    # chart = st.line_chart(last_rows)

    # total = len(data['train'])
    # for i, entry in enumerate(data['train']):
    #     _, new_row = tokenizer(entry['passages'][-1]['text'])
    # # for i in range(1, 101):
    #     # new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
    #     # status_text.text("%i%% Complete" % (i/total))
    #     chart.add_rows(np.array([float(len(new_row))]))
    #     progress_bar.progress(i/total)
    #     last_rows = new_row
    #     time.sleep(0.05)

    # progress_bar.empty()

    hist_data, labels = load_data(arg)
    fig = ff.create_distplot(hist_data, labels, bin_size=[0.1, 0.25, 0.5])
    st.title(f"Dataset stats for {arg}")

    st.plotly_chart(fig)
    # Streamlit widgets automatically run the script from top to bottom. Since
    # this button is not connected to any other logic, it just causes a plain
    # rerun.
    st.button("Re-run")
