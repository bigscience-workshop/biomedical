from distutils.command.build import build
import streamlit as st
import pandas as pd
import numpy as np
import glob
import datasets
from datasets import load_dataset

# data = load_dataset("biodatasets/<dataset_name>/<dataset_name>.py", name="<dataset_name>_bigbio_<schema>")
# import os

st.title('BigBIO datasets')

# Add a selectbox to the sidebar:

st.sidebar.header('Dataset Name')

file_paths = glob.glob('./biodatasets/*')

files = [i.split('/')[2] for i in file_paths]

dataset_name = st.sidebar.selectbox(
    'Name of dataset',
    tuple(files)
)

schema_select = st.sidebar.selectbox("Schema", ("Source Schema", "BigBio Schema"))


# This code snippet is from the datasets viewer to display the details in a clean way.
def render_features(features):
    if isinstance(features, dict):
        return {k: render_features(v) for k, v in features.items()}
    if isinstance(features, datasets.features.ClassLabel):
        return features.names

    if isinstance(features, datasets.features.Value):
        return features.dtype

    if isinstance(features, datasets.features.Sequence):
        return {"[]": render_features(features.feature)}
    return features


def load_ds(dataset_name):
    path = f"biodatasets/{dataset_name}/{dataset_name}.py"
    module_path = datasets.load.prepare_module(path, dataset=True)
    builder_cls = datasets.load.import_main_class(module_path[0], dataset=True)
    # st.write(builder_cls.BUILDER_CONFIGS[0].name, builder_cls.BUILDER_CONFIGS[1].name)
    
    # st.write(builder_cls.B
    st.write("Source Version: ", builder_cls.SOURCE_VERSION)
    # st.write("Homepage", builder_cls.HOMEPAGE)
    st.write("BigBio Version: ", builder_cls.BIGBIO_VERSION)
    # st.write("Supported Tasks", builder_cls.SUPPORTED_TASKS)
    if schema_select=="Source Schema":
        st.write("Source Schema: ", builder_cls.BUILDER_CONFIGS[0].schema)
        st.write("Source Subset ID: ", builder_cls.BUILDER_CONFIGS[0].subset_id)

        big_bio_data = load_dataset(path, name=builder_cls.BUILDER_CONFIGS[0].name)
        if st.checkbox("Show data splits"):
            st.write("Splits available: ", list(big_bio_data.keys()))


        val = str(list(big_bio_data.keys())[0])
        if st.checkbox('Show data types'):
            st.write("features: ", render_features(big_bio_data[val].features))

        st.write("Loading some sample data")
        st.write(big_bio_data[val][0:10])


    if schema_select=="BigBio Schema":
        st.write("BigBio Schema: ", builder_cls.BUILDER_CONFIGS[1].schema)
        st.write("BigBio Subset ID: ", builder_cls.BUILDER_CONFIGS[1].subset_id)

        big_bio_data = load_dataset(path, name=builder_cls.BUILDER_CONFIGS[1].name)
        if st.checkbox("Show data splits"):
            st.write("Splits available: ", list(big_bio_data.keys()))

        val = str(list(big_bio_data.keys())[0])
        if st.checkbox('Show data types'):
            st.write("features: ", render_features(big_bio_data[val].features))

        st.write("Loading some sample data")
        st.write(big_bio_data[val][0:10])


data = load_ds(dataset_name)
