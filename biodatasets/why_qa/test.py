import zipfile
import json

def read_zip_file(file_path):
    with zipfile.ZipFile(file_path) as zf:
        with zf.open("n2c2-community-annotations_2010-fan-why-QA/relations_whyqa_ann-v7-share.json") as f:
            dataset = json.load(f)
            return dataset


dataset = read_zip_file("C:/Users/franc/Desktop/n2c2-community-annotations_2010-fan-why-QA.zip")

samples = dataset['data'][0]['paragraphs']

for sample in samples:
    print(sample["qas"])
    print("######################################")





# for sample in samples:
#     for qa in sample['qas']:
#         print(qa['id'])
#         print(sample['note_id'])
#     print("######################################")
