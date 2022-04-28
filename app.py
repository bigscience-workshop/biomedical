from distutils.command.build import build
import streamlit as st
import pandas as pd
import numpy as np
import glob
import importlib
import datasets
from datasets import load_dataset


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



def task_subtypes():
    files = glob.glob("./biodatasets/*")
    task_dataset = {}
    for i in files:
        dataset_name = i.split('/')[-1]
        try:
            module = importlib.import_module(f'biodatasets.{dataset_name}.{dataset_name}')
            task_dataset[dataset_name] = module._SUPPORTED_TASKS
        except ModuleNotFoundError:
            continue

    all_tasks = []
    for k,v in task_dataset.items():
        all_tasks.append(v)
    all_tasks = sum(all_tasks, [])
    all_tasks = set(all_tasks)

    from collections import defaultdict
    result_dict = defaultdict(list)
    for key, value in task_dataset.items():
        for task_subset in task_dataset[key]:
            if task_subset in all_tasks:
                result_dict[task_subset].append(key)
    return result_dict

    



def load_ds(dataset_name):
    path = f"biodatasets/{dataset_name}/{dataset_name}.py"
    module = datasets.load.dataset_module_factory(path)
    builder_cls = datasets.load.import_main_class(module.module_path)
    st.write("Source Version: ", builder_cls.SOURCE_VERSION)
    st.write("BigBio Version: ", builder_cls.BIGBIO_VERSION)
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



results_dict = task_subtypes()
options = st.sidebar.selectbox(
     'Select Dataset By Task Type',
     results_dict.keys())


st.title('BigBIO datasets')

st.sidebar.header('Dataset Name')

dataset_name = st.sidebar.selectbox(
    'Name of dataset',
    tuple(results_dict[options])
)

schema_select = st.sidebar.selectbox("Schema", ("Source Schema", "BigBio Schema"))

data = load_ds(dataset_name)
