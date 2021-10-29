"""
n2c2 Obesity Recognition Challenge
Özlem Uzuner, PhD, Recognizing Obesity and Comorbidities in Sparse Data, Journal of the American Medical Informatics Association, Volume 16, Issue 4, July 2009, Pages 561–570, https://doi.org/10.1197/jamia.M3115
Usage:
python n2c2_obesity.py --indir data_dir/ --outdir output_dir/
"""
import os
import argparse
from pathlib import Path
from loguru import logger
from functools import partial
import xml.etree.ElementTree as et
from prompts import DatasetPrompts

_HOMEPAGE = "https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/"


def load_obesity_corpus(partition, data_dir, task="textual"):
    """
    Load the data split.
    :param partition: train/test
    :param data_dir: train and test data directory
    :param task: annotation type: texttual|intuitive; the textual task is used by default,
    since 532 examples in the intuitive task have unlabelled instances (low IAA)
    """
    logger.info(f"Loading n2c2 Obesity {partition} data")
    documents = {} #id : text
    all_diseases = set()
    notes = tuple()
    if partition == 'train':
        with open(data_dir / 'obesity_patient_records_training.xml') as t1, \
                open(data_dir / 'obesity_patient_records_training2.xml') as t2:
            notes1 = t1.read().strip()
            notes2 = t2.read().strip()
        notes = (notes1,notes2)
    elif partition == 'test':
        with open(data_dir / 'obesity_patient_records_test.xml') as t1:
            notes1 = t1.read().strip()
        notes = (notes1,)
        
    for file in notes:
        root = et.fromstring(file)
        root = root.findall("./docs")[0]
        for document in root.findall("./doc"):
            assert document.attrib['id'] not in documents
            documents[document.attrib['id']] = {}
            documents[document.attrib['id']]['text'] = document.findall("./text")[0].text
            
    annotation_files = tuple()
    if partition == 'train':
        with open(data_dir / 'obesity_standoff_annotations_training.xml') as t1, \
                open(data_dir / 'obesity_standoff_annotations_training_addendum.xml') as t2, \
                open(data_dir / 'obesity_standoff_annotations_training_addendum2.xml') as t3, \
                open(data_dir / 'obesity_standoff_annotations_training_addendum3.xml') as t4:
            train1 = t1.read().strip()
            train2 = t2.read().strip()
            train3 = t3.read().strip()
            train4 = t4.read().strip()
        annotation_files = (train1,train2,train3,train4)
    elif partition == 'test':
        with open(data_dir / 'obesity_standoff_annotations_test.xml') as t1:
            test1 = t1.read().strip()
        annotation_files = (test1,)

    for file in annotation_files:
        root = et.fromstring(file)
        for diseases_annotation in root.findall("./diseases"):

            annotation_source = diseases_annotation.attrib['source']
            assert isinstance(annotation_source, str)
            for disease in diseases_annotation.findall("./disease"):
                disease_name = disease.attrib['name']
                all_diseases.add(disease_name)
                for annotation in disease.findall("./doc"):
                    doc_id = annotation.attrib['id']
                    if not annotation_source in documents[doc_id]:
                        documents[doc_id][annotation_source] = {}
                    assert doc_id in documents
                    judgment = annotation.attrib['judgment']
                    documents[doc_id][annotation_source][disease_name] = judgment

    all_diseases = list(all_diseases)
    for id in documents: #set example to {} if contains unlabeled instance(s)
        for annotation_type in ('textual', 'intuitive'):
            for disease in all_diseases:
                if (not annotation_type in documents[id]) or (not disease in documents[id][annotation_type]):
                    documents[id][annotation_type] = {}

    lmap = {"Y": "present", "N": "absent", "U": "unmentioned", "Q": "questionable"}
    return [
        (
            id, documents[id]['text'], lmap[documents[id][task]["Obesity"]],
            lmap[documents[id][task]["Asthma"]], lmap[documents[id][task]["CAD"]], lmap[documents[id][task]["CHF"]],
            lmap[documents[id][task]["Depression"]], lmap[documents[id][task]["Diabetes"]], lmap[documents[id][task]["Gallstones"]],
            lmap[documents[id][task]["GERD"]], lmap[documents[id][task]["Gout"]], lmap[documents[id][task]["Hypercholesterolemia"]],
            lmap[documents[id][task]["Hypertension"]], lmap[documents[id][task]["Hypertriglyceridemia"]], lmap[documents[id][task]["OA"]],
            lmap[documents[id][task]["OSA"]], lmap[documents[id][task]["PVD"]], lmap[documents[id][task]["Venous Insufficiency"]],
        )
        for id in documents if documents[id][task]
    ]

class ObesityCorpusPrompts(DatasetPrompts):
    """
    Create prompts from n2c2 obesity corpus
    """
    def __init__(self, data_root):
        """Instantiate ObesityCorpusPrompts.
        :param data_root: Root folder containing target dataset
        """
        self.data_root = Path(data_root).resolve()
        self._init_dataset()
        self._init_prompts()

    def _init_dataset(self):
        """
        Prepare obesity data.
        Download data files listed below manually from https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/": 

        obesity_patient_records_training.xml
        obesity_patient_records_training2.xml
        obesity_standoff_annotations_training.xml
        obesity_standoff_annotations_training_addendum.xml
        obesity_standoff_annotations_training_addendum2.xml
        obesity_standoff_annotations_training_addendum3.xml
        obesity_patient_records_test.xml
        obesity_standoff_annotations_test.xml

        Downloading requires user registration and DUA.
        """
        self.splits = {
            split:load_obesity_corpus(split, self.data_root)
            for split in ['train', 'test']
        }
        logger.info("Dataset successfully loaded")

    def _init_prompts(self):
        """Initialize prompts."""
        self._prompts = {}
        self._metadata = {}


#
# Create Prompts + Helper Functions
#
def classify_obesity(x):
    """Simple prompt for detecting obesity.
    Args:
        x: a note from the obesity corpus.
    Returns:
        the prompt.
    """
    tmpl = '"{}"\n'
    tmpl += "Based on explicitly documented information in the above discharge summary, do you think the patient has obesity? Choose between 'present', 'absent', 'questionable' and 'unmentioned'.\n"
    tmpl += '|||{}'
    return tmpl.format(x[1], x[2])

def classify_asthma(x):
    """Simple prompt for detecting asthma.
    Args:
        x: a note from the obesity corpus.
    Returns:
        the prompt.
    """
    tmpl = '"{}"\n'
    tmpl += "Based on explicitly documented information in the above discharge summary, do you think the patient has asthma? Choose between 'present', 'absent', 'questionable' and 'unmentioned'.\n"
    tmpl += '|||{}'
    return tmpl.format(x[1], x[3])

def classify_CAD(x):
    """Simple prompt for detecting atherosclerotic cardiovascular disease (CAD).
    Args:
        x: a note from the obesity corpus.
    Returns:
        the prompt.
    """
    tmpl = '"{}"\n'
    tmpl += "Based on explicitly documented information in the above discharge summary, do you think the patient has atherosclerotic cardiovascular disease? Choose between 'present', 'absent', 'questionable' and 'unmentioned'.\n"
    tmpl += '|||{}'
    return tmpl.format(x[1], x[4])

def classify_CHF(x):
    """Simple prompt for detecting congestive heart failure (CHF).
    Args:
        x: a note from the obesity corpus.
    Returns:
        the prompt.
    """
    tmpl = '"{}"\n'
    tmpl += "Based on explicitly documented information in the above discharge summary, do you think the patient has congestive heart failure? Choose between 'present', 'absent', 'questionable' and 'unmentioned'.\n"
    tmpl += '|||{}'
    return tmpl.format(x[1], x[5])

def classify_depression(x):
    """Simple prompt for detecting depression.
    Args:
        x: a note from the obesity corpus.
    Returns:
        the prompt.
    """
    tmpl = '"{}"\n'
    tmpl += "Based on explicitly documented information in the above discharge summary, do you think the patient has depression? Choose between 'present', 'absent', 'questionable' and 'unmentioned'.\n"
    tmpl += '|||{}'
    return tmpl.format(x[1], x[6])

def classify_DM(x):
    """Simple prompt for detecting diabetes mellitus (DM).
    Args:
        x: a note from the obesity corpus.
    Returns:
        the prompt.
    """
    tmpl = '"{}"\n'
    tmpl += "Based on explicitly documented information in the above discharge summary, do you think the patient has diabetes mellitus? Choose between 'present', 'absent', 'questionable' and 'unmentioned'.\n"
    tmpl += '|||{}'
    return tmpl.format(x[1], x[7])

def classify_gallstones(x):
    """Simple prompt for detecting gallstones/cholecystectomy.
    Args:
        x: a note from the obesity corpus.
    Returns:
        the prompt.
    """
    tmpl = '"{}"\n'
    tmpl += "Based on explicitly documented information in the above discharge summary, do you think the patient has gallstones/cholecystectomy? Choose between 'present', 'absent', 'questionable' and 'unmentioned'.\n"
    tmpl += '|||{}'
    return tmpl.format(x[1], x[8])

def classify_GERD(x):
    """Simple prompt for detecting gastroesophageal reflux disease (GERD).
    Args:
        x: a note from the obesity corpus.
    Returns:
        the prompt.
    """
    tmpl = '"{}"\n'
    tmpl += "Based on explicitly documented information in the above discharge summary, do you think the patient has gastroesophageal reflux disease (GERD)? Choose between 'present', 'absent', 'questionable' and 'unmentioned'.\n"
    tmpl += '|||{}'
    return tmpl.format(x[1], x[9])

def classify_gout(x):
    """Simple prompt for detecting gout.
    Args:
        x: a note from the obesity corpus.
    Returns:
        the prompt.
    """
    tmpl = '"{}"\n'
    tmpl += "Based on explicitly documented information in the above discharge summary, do you think the patient has gout? Choose between 'present', 'absent', 'questionable' and 'unmentioned'.\n"
    tmpl += '|||{}'
    return tmpl.format(x[1], x[10])

def classify_hypercholesterolemia(x):
    """Simple prompt for detecting hypercholesterolemia.
    Args:
        x: a note from the obesity corpus.
    Returns:
        the prompt.
    """
    tmpl = '"{}"\n'
    tmpl += "Based on explicitly documented information in the above discharge summary, do you think the patient has hypercholesterolemia? Choose between 'present', 'absent', 'questionable' and 'unmentioned'.\n"
    tmpl += '|||{}'
    return tmpl.format(x[1], x[11])

def classify_HTN(x):
    """Simple prompt for detecting hypertension (HTN).
    Args:
        x: a note from the obesity corpus.
    Returns:
        the prompt.
    """
    tmpl = '"{}"\n'
    tmpl += "Based on explicitly documented information in the above discharge summary, do you think the patient has hypertension? Choose between 'present', 'absent', 'questionable' and 'unmentioned'.\n"
    tmpl += '|||{}'
    return tmpl.format(x[1], x[12])

def classify_hypertriglyceridemia(x):
    """Simple prompt for detecting hypertriglyceridemia.
    Args:
        x: a note from the obesity corpus.
    Returns:
        the prompt.
    """
    tmpl = '"{}"\n'
    tmpl += "Based on explicitly documented information in the above discharge summary, do you think the patient has hypertriglyceridemia? Choose between 'present', 'absent', 'questionable' and 'unmentioned'.\n"
    tmpl += '|||{}'
    return tmpl.format(x[1], x[13])

def classify_OSA(x):
    """Simple prompt for detecting obstructive sleep apnea (OSA).
    Args:
        x: a note from the obesity corpus.
    Returns:
        the prompt.
    """
    tmpl = '"{}"\n'
    tmpl += "Based on explicitly documented information in the above discharge summary, do you think the patient has obstructive sleep apnea? Choose between 'present', 'absent', 'questionable' and 'unmentioned'.\n"
    tmpl += '|||{}'
    return tmpl.format(x[1], x[14])

def classify_OA(x):
    """Simple prompt for detecting osteoarthritis (OA).
    Args:
        x: a note from the obesity corpus.
    Returns:
        the prompt.
    """
    tmpl = '"{}"\n'
    tmpl += "Based on explicitly documented information in the above discharge summary, do you think the patient has osteoarthritis? Choose between 'present', 'absent', 'questionable' and 'unmentioned'.\n"
    tmpl += '|||{}'
    return tmpl.format(x[1], x[15])

def classify_PVD(x):
    """Simple prompt for detecting peripheral vascular disease (PVD).
    Args:
        x: a note from the obesity corpus.
    Returns:
        the prompt.
    """
    tmpl = '"{}"\n'
    tmpl += "Based on explicitly documented information in the above discharge summary, do you think the patient has peripheral vascular disease? Choose between 'present', 'absent', 'questionable' and 'unmentioned'.\n"
    tmpl += '|||{}'
    return tmpl.format(x[1], x[16])

def classify_VI(x):
    """Simple prompt for detecting venous insufficiency (VI).
    Args:
        x: a note from the obesity corpus.
    Returns:
        the prompt.
    """
    tmpl = '"{}"\n'
    tmpl += "Based on explicitly documented information in the above discharge summary, do you think the patient has venous insufficiency ? Choose between 'present', 'absent', 'questionable' and 'unmentioned'.\n"
    tmpl += '|||{}'
    return tmpl.format(x[1], x[17])


def classify_obesity_question_first(x):
    """Simple prompt for detecting obesity with question first.
    Args:
        x: a note from the obesity corpus.
    Returns:
        the prompt.
    """
    tmpl = "Does the note below explicitly indicate the patient has obesity? Choose between 'present', 'absent', 'questionable' and 'unmentioned'.\n"
    tmpl += '"{}"\n|||{}'
    return tmpl.format(x[1], x[2])

def classify_asthma_question_first(x):
    """Simple prompt for detecting asthma with question first.
    Args:
        x: a note from the obesity corpus.
    Returns:
        the prompt.
    """
    tmpl = "Does the note below explicitly indicate the patient has asthma? Choose between 'present', 'absent', 'questionable' and 'unmentioned'.\n"
    tmpl += '"{}"\n|||{}'
    return tmpl.format(x[1], x[3])

def classify_CAD_question_first(x):
    """Simple prompt for detecting atherosclerotic cardiovascular disease (CAD) with question first.
    Args:
        x: a note from the obesity corpus.
    Returns:
        the prompt.
    """
    tmpl = "Does the note below explicitly indicate the patient has atherosclerotic cardiovascular disease? Choose between 'present', 'absent', 'questionable' and 'unmentioned'.\n"
    tmpl += '"{}"\n|||{}'
    return tmpl.format(x[1], x[4])

def classify_CHF_question_first(x):
    """Simple prompt for detecting congestive heart failure (CHF) with question first.
    Args:
        x: a note from the obesity corpus.
    Returns:
        the prompt.
    """
    tmpl = "Does the note below explicitly indicate the patient has congestive heart failure? Choose between 'present', 'absent', 'questionable' and 'unmentioned'.\n"
    tmpl += '"{}"\n|||{}'
    return tmpl.format(x[1], x[5])

def classify_depression_question_first(x):
    """Simple prompt for detecting depression with question first.
    Args:
        x: a note from the obesity corpus.
    Returns:
        the prompt.
    """
    tmpl = "Does the note below explicitly indicate the patient has depression? Choose between 'present', 'absent', 'questionable' and 'unmentioned'.\n"
    tmpl += '"{}"\n|||{}'
    return tmpl.format(x[1], x[6])

def classify_DM_question_first(x):
    """Simple prompt for detecting diabetes mellitus (DM) with question first.
    Args:
        x: a note from the obesity corpus.
    Returns:
        the prompt.
    """
    tmpl = "Does the note below explicitly indicate the patient has diabetes mellitus? Choose between 'present', 'absent', 'questionable' and 'unmentioned'.\n"
    tmpl += '"{}"\n|||{}'
    return tmpl.format(x[1], x[7])

def classify_gallstones_question_first(x):
    """Simple prompt for detecting gallstones/cholecystectomy with question first.
    Args:
        x: a note from the obesity corpus.
    Returns:
        the prompt.
    """
    tmpl = "Does the note below explicitly indicate the patient has gallstones/cholecystectomy? Choose between 'present', 'absent', 'questionable' and 'unmentioned'.\n"
    tmpl += '"{}"\n|||{}'
    return tmpl.format(x[1], x[8])

def classify_GERD_question_first(x):
    """Simple prompt for detecting gastroesophageal reflux disease (GERD) with question first.
    Args:
        x: a note from the obesity corpus.
    Returns:
        the prompt.
    """
    tmpl = "Does the note below explicitly indicate the patient has gastroesophageal reflux disease (GERD)? Choose between 'present', 'absent', 'questionable' and 'unmentioned'.\n"
    tmpl += '"{}"\n|||{}'
    return tmpl.format(x[1], x[9])

def classify_gout_question_first(x):
    """Simple prompt for detecting gout with question first.
    Args:
        x: a note from the obesity corpus.
    Returns:
        the prompt.
    """
    tmpl = "Does the note below explicitly indicate the patient has gout? Choose between 'present', 'absent', 'questionable' and 'unmentioned'.\n"
    tmpl += '"{}"\n|||{}'
    return tmpl.format(x[1], x[10])

def classify_hypercholesterolemia_question_first(x):
    """Simple prompt for detecting hypercholesterolemia with question first.
    Args:
        x: a note from the obesity corpus.
    Returns:
        the prompt.
    """
    tmpl = "Does the note below explicitly indicate the patient has hypercholesterolemia? Choose between 'present', 'absent', 'questionable' and 'unmentioned'.\n"
    tmpl += '"{}"\n|||{}'
    return tmpl.format(x[1], x[11])

def classify_HTN_question_first(x):
    """Simple prompt for detecting hypertension (HTN) with question first.
    Args:
        x: a note from the obesity corpus.
    Returns:
        the prompt.
    """
    tmpl = "Does the note below explicitly indicate the patient has hypertension? Choose between 'present', 'absent', 'questionable' and 'unmentioned'.\n"
    tmpl += '"{}"\n|||{}'
    return tmpl.format(x[1], x[12])

def classify_hypertriglyceridemia_question_first(x):
    """Simple prompt for detecting hypertriglyceridemia with question first.
    Args:
        x: a note from the obesity corpus.
    Returns:
        the prompt.
    """
    tmpl = "Does the note below explicitly indicate the patient has hypertriglyceridemia? Choose between 'present', 'absent', 'questionable' and 'unmentioned'.\n"
    tmpl += '"{}"\n|||{}'
    return tmpl.format(x[1], x[13])

def classify_OSA_question_first(x):
    """Simple prompt for detecting obstructive sleep apnea (OSA) with question first.
    Args:
        x: a note from the obesity corpus.
    Returns:
        the prompt.
    """
    tmpl = "Does the note below explicitly indicate the patient has obstructive sleep apnea? Choose between 'present', 'absent', 'questionable' and 'unmentioned'.\n"
    tmpl += '"{}"\n|||{}'
    return tmpl.format(x[1], x[14])

def classify_OA_question_first(x):
    """Simple prompt for detecting osteoarthritis (OA) with question first.
    Args:
        x: a note from the obesity corpus.
    Returns:
        the prompt.
    """
    tmpl = "Does the note below explicitly indicate the patient has osteoarthritis? Choose between 'present', 'absent', 'questionable' and 'unmentioned'.\n"
    tmpl += '"{}"\n|||{}'
    return tmpl.format(x[1], x[15])

def classify_PVD_question_first(x):
    """Simple prompt for detecting peripheral vascular disease (PVD) with question first.
    Args:
        x: a note from the obesity corpus.
    Returns:
        the prompt.
    """
    tmpl = "Does the note below explicitly indicate the patient has peripheral vascular disease? Choose between 'present', 'absent', 'questionable' and 'unmentioned'.\n"
    tmpl += '"{}"\n|||{}'
    return tmpl.format(x[1], x[16])

def classify_VI_question_first(x):
    """Simple prompt for detecting venous insufficiency (VI) with question first.
    Args:
        x: a note from the obesity corpus.
    Returns:
        the prompt.
    """
    tmpl = "Does the note below explicitly indicate the patient has venous insufficiency? Choose between 'present', 'absent', 'questionable' and 'unmentioned'.\n"
    tmpl += '"{}"\n|||{}'
    return tmpl.format(x[1], x[17])

def main(args):

    outpath = Path(args.outdir)
    dataset = ObesityCorpusPrompts(args.indir)

    # Add prompts
    prompts = {
        "classify_obesity":partial(classify_obesity),
        "classify_asthma":partial(classify_asthma),
        "classify_CAD":partial(classify_CAD),
        "classify_depression":partial(classify_depression),
        "classify_DM":partial(classify_DM),
        "classify_gallstones":partial(classify_gallstones),
        "classify_GERD":partial(classify_GERD),
        "classify_gout":partial(classify_gout),
        "classify_hypercholesterolemia":partial(classify_hypercholesterolemia),
        "classify_HTN":partial(classify_HTN),
        "classify_hypertriglyceridemia":partial(classify_hypertriglyceridemia),
        "classify_OSA":partial(classify_OSA),
        "classify_OA":partial(classify_OA),
        "classify_PVD":partial(classify_PVD),
        "classify_CHF":partial(classify_CHF),
        "classify_VI":partial(classify_VI),

        "classify_obesity_question_first":partial(classify_obesity_question_first),
        "classify_asthma_question_first":partial(classify_asthma_question_first),
        "classify_CAD_question_first":partial(classify_CAD_question_first),
        "classify_depression_question_first":partial(classify_depression_question_first),
        "classify_DM_question_first":partial(classify_DM_question_first),
        "classify_gallstones_question_first":partial(classify_gallstones_question_first),
        "classify_GERD_question_first":partial(classify_GERD_question_first),
        "classify_gout_question_first":partial(classify_gout_question_first),
        "classify_hypercholesterolemia_question_first":partial(classify_hypercholesterolemia_question_first),
        "classify_HTN_question_first":partial(classify_HTN_question_first),
        "classify_hypertriglyceridemia_question_first":partial(classify_hypertriglyceridemia_question_first),
        "classify_OSA_question_first":partial(classify_OSA_question_first),
        "classify_OA_question_first":partial(classify_OA_question_first),
        "classify_PVD_question_first":partial(classify_PVD_question_first),
        "classify_CHF_question_first":partial(classify_CHF_question_first),
        "classify_VI_question_first":partial(classify_VI_question_first),
    }
    for name in prompts:
        answers_in_prompt_ = True
        dataset.add_prompt(prompts[name],
                        name,
                        answer_keys=['present', 'absent', 'questionable', 'unmentioned'],
                        original_task=True,
                        answers_in_prompt=answers_in_prompt_,
                        metrics=['f1','accuracy'])

    df = dataset.get_prompts()

    # Save the dataset
    df.to_csv(outpath / 'obesity_prompts.tsv', sep='\t', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=str, default=None)
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()
    main(args)

