# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import List, Tuple, Dict

import datasets
from .bigbiohub import BigBioConfig
from bigbio.utils.constants import Tasks, Lang
from bigbio.utils.license import Licenses

import xml.etree.ElementTree as ET
import xmltodict
import collections

from .bigbiohub import kb_features

_LOCAL = True
_CITATION = """\
@article{10.1093/jamiaopen/ooab025,
    author = {Kittner, Madeleine and Lamping, Mario and Rieke, Damian T and Götze, Julian and Bajwa, Bariya and Jelas, Ivan and Rüter, Gina and Hautow, Hanjo and Sänger, Mario and Habibi, Maryam and Zettwitz, Marit and Bortoli, Till de and Ostermann, Leonie and Ševa, Jurica and Starlinger, Johannes and Kohlbacher, Oliver and Malek, Nisar P and Keilholz, Ulrich and Leser, Ulf},
    title = "{Annotation and initial evaluation of a large annotated German oncological corpus}",
    journal = {JAMIA Open},
    volume = {4},
    number = {2},
    year = {2021},
    month = {04},
    issn = {2574-2531},
    doi = {10.1093/jamiaopen/ooab025},
    url = {https://doi.org/10.1093/jamiaopen/ooab025},
    note = {ooab025},
    eprint = {https://academic.oup.com/jamiaopen/article-pdf/4/2/ooab025/38830128/ooab025.pdf},
}
"""
_DESCRIPTION = """\
BRONCO150 is a corpus containing selected sentences of 150 German discharge summaries of cancer patients (hepatocelluar carcinoma or melanoma) treated at Charite Universitaetsmedizin Berlin or Universitaetsklinikum Tuebingen. All discharge summaries were manually anonymized. The original documents were scrambled at the sentence level to make reconstruction of individual reports impossible.
"""
_HOMEPAGE = "https://www2.informatik.hu-berlin.de/~leser/bronco/index.html"
_LICENSE = Licenses.DUA
_URLS = {}
_PUBMED = False
_SUPPORTED_TASKS = [Tasks.NAMED_ENTITY_RECOGNITION, Tasks.NAMED_ENTITY_DISAMBIGUATION]
_SOURCE_VERSION = "1.0.0"
_BIGBIO_VERSION = "1.0.0"
_DATASETNAME = "bronco"
_DISPLAYNAME = "BRONCO"
_LANGUAGES = [Lang.DE]


class Bronco(datasets.GeneratorBasedBuilder):
    SOURCE_VERSION = datasets.Version(_SOURCE_VERSION)
    BIGBIO_VERSION = datasets.Version(_BIGBIO_VERSION)
    DEFAULT_CONFIG_NAME = "bronco_bigbio_kb"

    # ds_bigbio = datasets.load_dataset('my_dataset', name='bigbio', data_dir="/path/to/data/files")

    BUILDER_CONFIGS = [
        BigBioConfig(
            name="bronco_source",
            version=SOURCE_VERSION,
            description="BRONCO source schema",
            schema="source",
            subset_id="bronco",
        ),
        BigBioConfig(
            name="bronco_bigbio_kb",
            version=BIGBIO_VERSION,
            description="BRONCO BigBio schema",
            schema="bigbio_kb",
            subset_id="bronco",
        ),
    ]

    DEFAULT_CONFIG_NAME = "[dataset_name]_source"

    def _info(self) -> datasets.DatasetInfo:

        # Create the source schema; this schema will keep all keys/information/labels as close to the original dataset as possible.

        # You can arbitrarily nest lists and dictionaries.
        # For iterables, use lists over tuples or `datasets.Sequence`

        if self.config.schema == "source":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "passage": {
                        "offset": datasets.Value("string"),
                        "text": datasets.Value("string"),
                        "annotation": [
                            {
                                "@id": datasets.Value("string"),
                                "infon": [
                                    {
                                        "@key": datasets.Value("string"),
                                        "#text": datasets.Value("string"),
                                    }
                                ],
                                "location": [
                                    {
                                        "@offset": datasets.Value("string"),
                                        "@length": datasets.Value("string"),
                                    }
                                ],
                                "text": datasets.Value("string"),
                            }
                        ],
                        "relation": [
                            {
                                "@id": datasets.Value("string"),
                                "infon": [
                                    {
                                        "@key": datasets.Value("string"),
                                        "#text": datasets.Value("string"),
                                    }
                                ],
                                "node": {
                                    "@refid": datasets.Value("string"),
                                    "@role": datasets.Value("string"),
                                },
                            }
                        ],
                    }
                }
            )

        elif self.config.schema == "bigbio_kb":
            features = kb_features

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        if self.config.data_dir is None:
            raise ValueError("This is a local dataset. Please pass the data_dir kwarg to load_dataset.")
        else:
            data_dir = self.config.data_dir
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "bioCFiles", "BRONCO150.xml"),
                    "split": "train",
                },
            ),
        ]

    def _generate_examples(self, filepath, split: str) -> Tuple[int, Dict]:
        """Yields examples as (key, example) tuples."""

        with open(filepath, encoding="utf8") as file:

            tree = ET.parse(file)
            xml_data = tree.getroot()
            xmlstr = ET.tostring(xml_data, encoding='utf-8', method='xml')
            data = xmltodict.parse(xmlstr)
            data = data['collection']['document']

            if self.config.schema == "source":

                for uid, doc in enumerate(data):
                    out = {
                        'id': doc['id'],
                        'passage': {
                            'offset': doc['passage']['offset'],
                            'text': doc['passage']['text'],
                            'annotation': [],
                            'relation': [],
                        }
                    }

                    for annotation in doc['passage']['annotation']:
                        anno = {
                            '@id': annotation['@id'],
                            'infon': [],
                            'text': annotation['text'],
                        }

                        for infon in annotation['infon']:
                            anno['infon'].append({
                                '@key': infon['@key'],
                                '#text': infon['#text'],
                            })

                        if isinstance(annotation['location'], list):
                            split_location = []
                            for location in annotation['location']:
                                split_location.append({
                                    '@offset': location['@offset'],
                                    '@length': location['@length'],
                                })
                            anno['location'] = split_location
                        elif isinstance(annotation['location'], collections.OrderedDict):
                            anno['location'] = [{
                                '@offset': location['@offset'],
                                '@length': location['@length'],
                            }]

                        out['passage']['annotation'].append(anno)

                    for relation in doc['passage']['relation']:
                        rel = {
                            "@id": relation["@id"],
                            "infon": [],
                        }

                        for infon in relation['infon']:
                            rel['infon'].append({
                                '@key': infon['@key'],
                                '#text': infon['#text'],
                            })

                        rel['node'] = {
                            '@refid': relation['node']['@refid'],
                            '@role': relation['node']['@role'],
                        }
                        out['passage']['relation'].append(rel)

                    yield uid, out

            elif self.config.schema == "bigbio_kb":
                for uid, doc in enumerate(data):
                    out = {
                        'id': uid,
                        'document_id': doc['id'],
                        'passages': [],
                        'entities': [],
                        'events': [],
                        'coreferences': [],
                        'relations': [],
                    }

                    # catch all normalized entities
                    norm_map = {}
                    for rel in doc['passage']['relation']:
                        if rel['infon'][-1]['#text'] == 'Normalization':
                            norm_map[rel['node']['@role']] = rel['node']['@refid']

                    # pass sentences to passages
                    for i, passage in enumerate(doc['passage']['text'].split('\n')):
                        # match the offsets on the text after removing \n
                        if i == 0:
                            marker = 0
                        else:
                            marker = out['passages'][-1]['offsets'][-1][-1] + 1

                        out['passages'].append({
                            'id': f'{uid}-{i}',
                            'text': [passage],
                            'type': 'sentence',
                            'offsets': [[marker, marker + len(passage)]],
                        })

                    # handle entities
                    for ent in doc['passage']['annotation']:
                        offsets = []
                        text_s = []
                        # handle entities splitted across the text
                        if isinstance(ent['location'], list):
                            for part in ent['location']:
                                start = int(part['@offset'])
                                end = int(part['@offset']) + int(part['@length'])
                                offsets.append([start, end])
                                text_s.append(doc['passage']['text'][start:end])
                        # handle entities where all characters are consecutive
                        elif isinstance(ent['location'], collections.OrderedDict):
                            start = int(ent['location']['@offset'])
                            end = int(ent['location']['@offset']) + int(ent['location']['@length'])
                            offsets.append([start, end])
                            text_s.append(ent['text'])

                        out['entities'].append({
                            'id': f'{uid}-{ent["@id"]}',
                            'type': ent['infon'][1]['#text'],
                            'text': text_s,
                            'offsets': offsets,
                            'normalized': [{
                                'db_name': norm_map.get(ent["@id"], ':').split(':')[0],
                                # replace faulty connectors in db_ids
                                'db_id': norm_map.get(ent["@id"], ':').split(':')[1].replace(",",".").replace("+",""),
                            }]
                        })

                    yield uid, out


if __name__ == "__main__":
    datasets.load_dataset(path = 'bronco.py', data_dir=r"C:\Users\admin\Desktop\BRONCO150", name="bronco_source")
