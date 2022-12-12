#!/usr/bin/env python

"""
Script to automate hub migration.
Assumes `bigbiohub.py` script is present
"""
import argparse
from bigbio.utils.constants import Lang
from bigbio.utils.license import Licenses

from typing import List

lang_dict = {
    key: getattr(Lang, key).value for key in dir(Lang) if "__" not in key
}

license_dict = {
    key: getattr(Licenses, key).name
    for key in dir(Licenses)
    if "__" not in key
}


def fix_script(script: List[str]) -> List[str]:
    """Converts all bigbio references to the bigbio hub
    references
    """
    # Get task type
    ft_type = get_task_features(script)

    # Get license line
    license = get_license(script)

    # Get Language
    language = get_language(script)

    bblines = [
        "from .bigbiohub import " + ft_type + "\n",
        "from .bigbiohub import BigBioConfig\n"
        "from .bigbiohub import Tasks\n",
    ]

    # Make changes within the lines
    new_lines = []
    start_idx = []
    for idx, line in enumerate(script):
        if "bigbio.utils" in line:
            start_idx.append(idx)
        elif "_LANGUAGES" in line:
            new_lines.append("_LANGUAGES = [" + language + "]\n")

        # NOTE: this might fail if _LICENSE is used without spacing... so i checked
        # with also the Licenses. call

        elif ("_LICENSE" in line) and ("Licenses." in line):
            new_lines.append("_LICENSE = '" + license + "'\n")
        elif "features = schemas." in line:
            new_lines.append(line.replace("schemas.", ""))
        else:
            new_lines.append(line)

    new_lines = (
        new_lines[: start_idx[0]] + bblines + new_lines[start_idx[0] :]
    )

    return new_lines


def get_task_features(lines: List[str]) -> str:
    """Get the feature type for schema"""
    ft_type = [i.split("schemas.")[1] for i in lines if "schemas." in i]
    if len(ft_type) < 1:
        raise ValueError("Schema feature not identified")
    else:
        ft_type = ft_type[0].replace("\n", "")

    return ft_type


def get_license(lines: List[str]) -> str:
    """Get the license information"""
    license = [
        i.split("Licenses.")[1].replace("\n", "")
        for i in lines
        if "_LICENSE = " in i
    ]

    if len(license) != 1:
        raise ValueError("License not found")
    else:
        license = license_dict[license[0]]

    return license


def get_language(lines: List[str]) -> str:
    """Get the language information"""
    language = [
        i.split("[")[1].split("]")[0] for i in lines if "_LANGUAGES" in i
    ]
    if len(language) != 1:
        raise ValueError("Language not found")
    else:
        language = language[0].split(",")
        language = [lang_dict[i.split("Lang.")[1]] for i in language]
        language = ["'" + str(i) + "'" for i in language]

        if len(language) == 1:
            return language[0]
        else:
            return ", ".join(language)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--script", help="path to dataloading script", required=True
    )
    parser.add_argument(
        "--savename",
        help="path to dataloading script",
    )

    args = parser.parse_args()

    if args.savename is None:
        args.savename = (
            args.script.split(".py")[0].split("/")[-1] + "_hub.py"
        )

    with open(args.script, "r") as f:
        scriptlines = f.readlines()

    newlines = fix_script(scriptlines)

    with open(args.savename, "w") as f:
        f.writelines(newlines)
