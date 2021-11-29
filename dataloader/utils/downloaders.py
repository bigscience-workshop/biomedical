import os
from glob import glob

import gzip
import tarfile
import zipfile
import urllib.request

import re
import argparse
from typing import Dict

# -------------------------- #
# Generic downloading functions
# -------------------------- #


def download_data(url: str, fpath):
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-agent", "Mozilla/5.0")]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, fpath)


def uncompress_data(fpath, outfpath):
    ext = os.path.os.path.splitext(fpath)
    if ext[-1] == ".zip":
        with zipfile.ZipFile(fpath, "r") as zip_ref:
            # HACK for files containing MacOS garbage files
            for zobj in zip_ref.namelist():
                if "__MACOSX" in zobj:
                    continue
                zip_ref.extract(zobj, path=outfpath)
            # zip_ref.extractall(outfpath)


# -------------------------- #
# Chemprot specific helpers
# -------------------------- #
"""
2020.10.10

Chemprot helper functions to extract relevant information.
The following script converts the ChemProt dataset into a Standoff format for easier processing.

There are several components to the ChemProt dataset:

Note, for Relations in particular, there is a GROUP annotation Y/N - it might be worth filtering for ONLY those that have "Y" as this was the explicitly criteria used for evaluations.

(1) Abstracts - the actual natural text (PMID, Title, Abstract)
(2) Entities - a list of entities (PMID, Entity, Type of entity, start char, end char, text of entity)
(3) Relations (PMID, Chem Protein relation CPR group, Eval Type, CPR Type, Interactor Arg 1, Interactor Arg 2)
(4) Gold Standard Relations - Similar to (3) but manually annotated and confirmed.

The relations should be explicitly the "Gold" standard, as this was explicitly manually annotated (hence "ground truth")

This script is largely based on John Giogri's version found here:
https://github.com/JohnGiorgi/ChemProt-to-Standoff/blob/master/chemprot_to_standoff.py

Usage:

python utils_chemprot -i /dir/to/chemprot/corpus/datasplit -o output_dir
"""


def chemprot_2_standoff(data_dir: str, output_dir: str):
    """
    Convert the ChemProt dataset into a BRAT-Standoff format.

    :param data_dir: Unzipped ChemProt corpus files directory
    :param output_dir: Location to store data
    """
    if not os.path.exists(output_dir):
        print("Output dir does not exist, making it")  # TODO convert to log
        os.makedirs(output_dir)

    for filepath in glob(os.path.join(data_dir, "*.tsv")):

        # Get abstracts
        if re.search("abstract", filepath):
            abstracts = get_abstract(filepath)

            # Create txt files
            write_data(abstracts, output_dir, "txt")

        # Get entities
        elif re.search("entities", filepath):
            ents = get_entities(filepath)

        elif re.search("gold_standard", filepath):
            relns = get_relations(filepath)

    # Concatenate the Entities + Relations files together
    for pmid in ents:
        if pmid in relns:
            ents[pmid] = f"{ents[pmid]}\n{relns[pmid]}"

    write_data(ents, output_dir, "ann")


def write_data(data_dict: Dict[str, str], output_dir: str, ext: str = "txt"):
    """
    For each PMID, save the record to txt or ann file.

    :param data_dict: Dictionary of abstracts/entities/relations
    :param output_dir: Directory to save data in
    :param ext: Type of file extension (text for abstracts and ann for else)

    :returns: None; Prints the dictionary, per PMID, to file.
    """
    for pmid, item in data_dict.items():

        filename = os.path.join(output_dir, f"{pmid}.{ext}")

        with open(filename, "w") as f:
            f.write(item)


def get_abstract(abs_filename: str) -> Dict[str, str]:
    """
    For each document in PubMed ID (PMID) in the ChemProt abstract data file, return the abstract. Data is tab-separated.

    :param filename: `*_abstracts.tsv from ChemProt

    :returns Dictionary with PMID keys and abstract text as values.
    """
    with open(abs_filename, "r") as f:
        contents = [i.strip() for i in f.readlines()]

    # PMID is the first column, Abstract is last
    return {
        doc.split("\t")[0]: "\n".join(doc.split("\t")[1:]) for doc in contents
    }  # Includes title as line 1


def get_entities(ents_filename: str) -> Dict[str, str]:
    """
    For each document in the corpus, return entity annotations per PMID.
    Each column in the entity file is as follows:
    (1) PMID
    (2) Entity Number
    (3) Entity Type (Chemical, Gene-Y, Gene-N)
    (4) Start index
    (5) End index
    (6) Actual text of entity

    :param ents_filename: `_*entities.tsv` file from ChemProt

    :returns: Dictionary with PMID keys and entity annotations.
    """
    with open(ents_filename, "r") as f:
        contents = [i.strip() for i in f.readlines()]

    entities = {}

    for line in contents:

        pmid, idx, label, start_offset, end_offset, name = line.split("\t")

        # If no PMID in dict, add empty container
        if pmid not in entities:
            entities[pmid] = []

        # If PMID already available, add new entities
        ann = f"{idx}\t{label} {start_offset} {end_offset}\t{name}"
        entities[pmid].append(ann)

    return {pmid: "\n".join(ann) for pmid, ann in entities.items()}


def get_relations(rel_filename: str) -> Dict[str, str]:
    """
    For each document in the ChemProt corpus, create an annotation for the gold-standard relationships.

    The columns include:
    (1) PMID
    (2) Relationship Label (CPR)
    (3) Interactor Argument 1 + Identifier
    (4) Interactor Argument 2 + Identifier

    Gold standard includes CPRs 3-9. Relationships are always Gene + Protein.
    Unlike entities, there is no counter, hence once must be made

    :param rel_filename: Gold standard file name
    """
    with open(rel_filename, "r") as f:
        contents = [i.strip() for i in f.readlines()]

    ridx = 1  # Counter for relations index
    relations = {}

    for line in contents:
        pmid, label, arg1, arg2 = line.split("\t")

        if pmid not in relations:
            ridx = 1
            relations[pmid] = []

        ann = f"R{ridx}\t{label} {arg1} {arg2}"
        relations[pmid].append(ann)

        ridx += 1

    return {pmid: "\n".join(ann) for pmid, ann in relations.items()}


# Borrowed from:
# https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist#273227
def make_dir(directory):
    """
    Creates a directory at `directory` if it does not already exist.
    """
    try:
        os.makedirs(directory)
    except OSError as err:
        if err.errno != errno.EEXIST:
            raise


# Note in the ChemProt dataset, the gold standard is generally equal to the relations with manual annotation as a group labeled as "Y"

# import pandas as pd
#
# for i in ["_development", "_sample", "_training", "_test_gs"]:
#    file1 = 'chemprot' + i + '/chemprot' + i + '_relations.tsv'
#    file2 = 'chemprot' + i + '/chemprot' + i + '_gold_standard.tsv'
#    x = pd.read_csv(file1, sep="\t", header=None)
#    y = pd.read_csv(file2, sep="\t", header=None)
#
#    x.loc[:, 2] = x.loc[:, 2].apply(lambda x: x.split()[0])
#    print("\n\n" + i)
#    print("Relations with Y=", x[x.loc[:, 2]=="Y"].shape[0])
#    print("Gold Standard=", y.shape[0])
