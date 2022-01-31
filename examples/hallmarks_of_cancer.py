import os 
import datasets
import glob


_DATASETNAME = "hallmarks_of_cancer"

_CITATION = """\
@article{baker2015automatic,
  title={Automatic semantic classification of scientific literature according to the hallmarks of cancer},
  author={Baker, Simon and Silins, Ilona and Guo, Yufan and Ali, Imran and H{\"o}gberg, Johan and Stenius, Ulla and Korhonen, Anna},
  journal={Bioinformatics},
  volume={32},
  number={3},
  pages={432--440},
  year={2015},
  publisher={Oxford University Press}
}
"""

_DESCRIPTION = """\
The Hallmarks of Cancer (HOC) Corpus consists of 1852 PubMed publication abstracts 
manually annotated by experts according to a taxonomy. The taxonomy consists of 37 
classes in a hierarchy. Zero or more class labels are assigned to each sentence in 
the corpus. The labels are found under the "labels" directory, while the tokenized 
text can be found under "text" directory. The filenames are the corresponding PubMed IDs (PMID).
"""

_HOMEPAGE = "https://github.com/sb895/Hallmarks-of-Cancer"

_LICENSE = "GNU General Public License v3.0"

_URLs = {"hallmarks_of_cancer": "https://github.com/sb895/Hallmarks-of-Cancer/archive/refs/heads/master.zip"}

_VERSION = "1.0.0"


_CLASS_NAMES = [
    'Activating invasion and metastasis',
    'Avoiding immune destruction',
    'Cellular energetics',
    'Enabling replicative immortality',
    'Evading growth suppressors',
    'Genomic instability and mutation',
    'Inducing angiogenesis',
    'Resisting cell death',
    'NULL',
    'Sustaining proliferative signaling',
    'Tumor promoting inflammation'
]


class Hallmarks_Of_Cancer(datasets.GeneratorBasedBuilder):
    """Hallmarks Of Cancer Dataset"""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=_DATASETNAME,
            version=VERSION,
            description=_DESCRIPTION,
        ),
    ]

    DEFAULT_CONFIG_NAME = (
        _DATASETNAME  # It's not mandatory to have a default configuration. Just use one if it make sense.
    )

    def _info(self):

        if self.config.name == _DATASETNAME:
            features = datasets.Features(
                {
                "text": datasets.Value("string"),
                "labels": datasets.Sequence(datasets.ClassLabel(names=_CLASS_NAMES)),
                "id": datasets.Value("string"),
                }
            )

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLs
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        data_dir = dl_manager.download_and_extract(_URLs)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": data_dir,
                    "split": "train",
                }
            )
        ]

    def _generate_examples(self, filepath, split):

        path_name = list(filepath.values())[0]+'/*'
        texts = glob.glob(path_name + '/text/*')
        labels = glob.glob(path_name + '/labels/')
        key = 0

        if self.config.name == _DATASETNAME:
            for tf_name in texts:
                filenname = os.path.basename(tf_name)
                with open(tf_name, encoding='utf-8') as f:
                    lines = f.readlines()
                    text_body = "".join([j.strip() for j in lines])

                label_file_name = labels[0] + '/' + filenname
                with open(label_file_name, encoding='utf-8') as f:
                    lines = f.readlines()
                    label_body = "".join([j.strip() for j in lines])
                    label_body = [i.strip() for i in label_body.split('<')]
                    label_body = sum([k.split('AND') for k in label_body if len(k)>1], [])
                    label_body = [i.split('--')[0].strip() for i in label_body]
                
                yield key, {
                                'id': filenname.split('.')[0],
                                'text': text_body,
                                'labels': label_body
                            
                            }

                key += 1


