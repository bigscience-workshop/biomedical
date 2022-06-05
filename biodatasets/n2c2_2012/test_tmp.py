import tarfile
from collections import defaultdict, OrderedDict
import os
from unittest import skip
from lxml import etree
import xmltodict
import json

"""""
def _read_tar_gz_old_(file_path, samples=None):
    if samples is None:
        samples = defaultdict(dict)
    print(samples)
    with tarfile.open(file_path, "r:gz") as tf:
        for member in tf.getmembers():

            base, filename = os.path.split(member.name)
            _, ext = os.path.splitext(filename)
            ext = ext[1:]  # get rid of dot
            sample_id = filename.split(".")[0]

            if ext == "xml" and not filename in ["23.xml", "143.xml", "152.xml", "272.xml","382.xml","422.xml","547.xml","807.xml"]:
                with tf.extractfile(member) as fp:
                    content_bytes = fp.read()
                content = content_bytes.decode("utf-8").encode()
                root = etree.XML(content)
                text, tags = root.getchildren()
                samples[sample_id]["txt"] = text.text
                samples[sample_id]["tags"] = {}

                for child in tags:
                          

                    if child.tag == "EVENT":
                        samples[sample_id]["tags"][child.tag]["id"] = child.get("id")
                        samples[sample_id]["tags"][child.tag]["start"] = child.get("start")
                        samples[sample_id]["tags"][child.tag]["end"] = child.get("end")
                        samples[sample_id]["tags"][child.tag]["text"] = child.get("text")
                        samples[sample_id]["tags"][child.tag]["modality"] = child.get("modality")
                        samples[sample_id]["tags"][child.tag]["polarity"] = child.get("polarity")
                        samples[sample_id]["tags"][child.tag]["type"] = child.get("type")
                    if child.tag == "TIMEx3":
                        samples[sample_id]["tags"][child.tag]["id"] = child.get("id")
                        samples[sample_id]["tags"][child.tag]["start"] = child.get("start")
                        samples[sample_id]["tags"][child.tag]["end"] = child.get("end")
                        samples[sample_id]["tags"][child.tag]["text"] = child.get("text")
                        samples[sample_id]["tags"][child.tag]["type"] = child.get("type")
                        samples[sample_id]["tags"][child.tag]["val"] = child.get("val")
                        samples[sample_id]["tags"][child.tag]["mod"] = child.get("mod")
                    if child.tag == "TLINK":
                        samples[sample_id]["tags"][child.tag]["id"] = child.get("id")
                        samples[sample_id]["tags"][child.tag]["fromID"] = child.get("fromID")
                        samples[sample_id]["tags"][child.tag]["fromText"] = child.get("fromTExt")
                        samples[sample_id]["tags"][child.tag]["toID"] = child.get("toID")
                        samples[sample_id]["tags"][child.tag]["toText"] = child.get("toText")
                        samples[sample_id]["tags"][child.tag]["type"] = child.get("type")
                    if child.tag == "SECTIME":
                        samples[sample_id]["tags"][child.tag]["id"] = child.get("id")
                        samples[sample_id]["tags"][child.tag]["start"] = child.get("start")
                        samples[sample_id]["tags"][child.tag]["end"] = child.get("end")
                        samples[sample_id]["tags"][child.tag]["text"] = child.get("text")
                        samples[sample_id]["tags"][child.tag]["type"] = child.get("type")
                        samples[sample_id]["tags"][child.tag]["dvalue"] = child.get("dvalue")

"""""

def _read_tar_gz_train_(file_path, samples=None):
    if samples is None:
        samples = defaultdict(dict)
    print(samples)
    with tarfile.open(file_path, "r:gz") as tf:
        for member in tf.getmembers():

            base, filename = os.path.split(member.name)
            _, ext = os.path.splitext(filename)
            ext = ext[1:]  # get rid of dot
            sample_id = filename.split(".")[0]

            if ext == "xml" and not filename in ["23.xml", "143.xml", "152.xml", "272.xml","382.xml","422.xml","547.xml","807.xml"]:
                with tf.extractfile(member) as fp:
                    content_bytes = fp.read()
                content = content_bytes.decode("utf-8").encode()
                values = xmltodict.parse(content)
                samples[sample_id] = values["ClinicalNarrativeTemporalAnnotation"] 

    samples_sorted = OrderedDict(sorted(samples.items(), key=lambda x: int(x[0])))
    samples = samples_sorted
    samples = json.loads(json.dumps(samples))

    return samples            

    """
    with open('C:/Users/franc/Desktop/result.json', 'w') as fp:
        json.dump(samples, fp)


    for i, event in enumerate(samples["1"]["TAGS"]["EVENT"]):
        print(event["@id"])
        #print(samples["1"]["TAGS"]["EVENT"][event])

    admission = {}
    discharge = {}
    for idx, sectime in enumerate(samples["1"]["TAGS"]["SECTIME"]):
        if sectime["@type"] == "ADMISSION":
            admission = {
                "id": sectime["@id"],
                "type": sectime["@type"],
                "text": sectime["@text"],
                "offsets": [(sectime["@start"], sectime["@end"])],
                }
        elif sectime["@type"] == "DISCHARGE":
            discharge = {
                "id": sectime["@id"],
                "type": sectime["@type"],
                "text": sectime["@text"],
                "offsets": [(sectime["@start"], sectime["@end"])],
            }
        print(admission)

    
    sample = samples["1"]
    x = {"id": 1, "tags": {
                "EVENT": sample["TAGS"]["EVENT"]}
        }
    print(x)

    for sample_id, sample in samples.items():
        print(sample)
        print("/////")
        print(sample_id)
        print("-----------------------------------------------------------------------------")

    """
########################################################################################################################
def _read_tar_gz_test_(file_path, samples=None):
    if samples is None:
        samples = defaultdict(dict)
    print(samples)
    with tarfile.open(file_path, "r:gz") as tf:
        for member in tf.getmembers():
            if member.name.startswith("ground_truth/merged_xml"):

                base, filename = os.path.split(member.name)
                _, ext = os.path.splitext(filename)
                ext = ext[1:]  # get rid of dot
                sample_id = filename.split(".")[0]

                if ext == "xml" and not filename in ["53.xml", "397.xml","527.xml","627.xml","687.xml","802.xml"]:
                    with tf.extractfile(member) as fp:
                        content_bytes = fp.read()
                    content = content_bytes.decode("utf-8").encode()
                    values = xmltodict.parse(content)
                    samples[sample_id] = values["ClinicalNarrativeTemporalAnnotation"] 

    samples_sorted = OrderedDict(sorted(samples.items(), key=lambda x: int(x[0])))
    samples = samples_sorted

    return samples

    #with open('C:/Users/franc/Desktop/result_test.json', 'w') as fp:
        #json.dump(samples, fp)

############################################################################################################################################################################
def  _get_events_from_sample(sample_id, sample):
    events = []
    for idx, event in enumerate(sample["TAGS"]["EVENT"]):
        
        evs = {
        "id": event["@id"],
        "type": event["@type"],
        "offsets": [(event["@start"], event["@end"])],
        "text":  event["@text"],
         }

        events.append(evs)
    print(events)

def _get_source_sample(sample_id, sample):
    output = {
        "id": sample_id,
        "text": sample["TEXT"],
        "tags": sample["TAGS"],
        }
    return output

   
#_read_tar_gz_train_("C:/Users/franc/Desktop/2012-07-15.original-annotation.release.tar.gz")
#_read_tar_gz_test_("C:/Users/franc/Desktop/2012-08-23.test-data.groundtruth.tar.gz")


def test():
    samples = _read_tar_gz_train_("C:/Users/franc/Desktop/2012-07-15.original-annotation.release.tar.gz")
    #samples = _read_tar_gz_test_("C:/Users/franc/Desktop/2012-08-23.test-data.groundtruth.tar.gz")

    #print(samples)
    _id = 0
    for sample_id, sample in samples.items():
        if sample.get("TAGS","").get("SECTIME","") == "":
            print("empty")
        else:
            print(sample.get("TAGS","").get("SECTIME",""))
        print(_id)
        print(sample_id)
        print("-----------------------------------------------------------------------------")
        _id += 1


def test_2_():
    samples = _read_tar_gz_train_("C:/Users/franc/Desktop/2012-07-15.original-annotation.release.tar.gz")

    for sample_id, sample in samples.items():
        events = []

        for idx, event in enumerate(sample.get("TAGS","").get("EVENT","")):
            
            evs = {
            "id": event.get("@id",""),
            "type": event.get("@type",""),
            "trigger": {
                "text": event.get("@text",""),
                "offests": [(int(event.get("@start","")), int(event.get("@end","")))],
                },
            "arguments": [
                {
                "role": "NA",
                "ref_id": "NA",
                },
            ],
            }
            events.append(evs)
    
        print(events)
        print("############################################################")
        print("############################################################")

def test_3_():
    samples = _read_tar_gz_train_("C:/Users/franc/Desktop/2012-07-15.original-annotation.release.tar.gz")

    for sample_id, sample in samples.items():
        print(sample_id)
        print(len(sample.get("TAGS","").get("SECTIME","")))

        admission = []

        if sample.get("TAGS","").get("SECTIME","") == "":
            pass
        elif len(sample.get("TAGS","").get("SECTIME","")) == 2:
            for idx, sectime in enumerate(sample.get("TAGS","").get("SECTIME","")):
                if sectime.get("@type","") == "ADMISSION":
                    adm = {
                        "id": sectime.get("@id",""),
                        "type": sectime.get("@type",""),
                        "text": [sectime.get("@text","")],
                        "offsets": [(int(sectime.get("@start","")), int(sectime.get("@end","")))],
                        }
                    admission.append(adm)
        else:
            sectime = sample.get("TAGS","").get("SECTIME","")
            if sectime.get("@type","") == "ADMISSION":
                adm = {
                    "id": sectime.get("@id",""),
                    "type": sectime.get("@type",""),
                    "text": [sectime.get("@text","")],
                    "offsets": [(int(sectime.get("@start","")), int(sectime.get("@end","")))],
                    }
                admission.append(adm) 

        print(admission)
        print("############################################################")


def test_4_():
    samples = _read_tar_gz_train_("C:/Users/franc/Desktop/2012-07-15.original-annotation.release.tar.gz")

    for sample_id, sample in samples.items():
        print(sample_id)
        print(len(sample.get("TAGS","").get("SECTIME","")))

        discharge = []
        
        if sample.get("TAGS","").get("SECTIME","") == "":
            pass
        elif len(sample.get("TAGS","").get("SECTIME","")) == 2:
            for idx, sectime in enumerate(sample.get("TAGS","").get("SECTIME","")):
                if sectime.get("@type","") == "DISCHARGE":
                    dis = {
                        "id": sectime.get("@id",""),
                        "type": sectime.get("@type",""),
                        "text": [sectime.get("@text","")],
                        "offsets": [(sectime.get("@start",""), sectime.get("@end",""))],
                        }
                    discharge.append(dis)
        else:
            sectime = sample.get("TAGS","").get("SECTIME","")
            if sectime.get("@type","") == "DISCHARGE":
                dis = {
                    "id": sectime.get("@id",""),
                    "type": sectime.get("@type",""),
                    "text": [sectime.get("@text","")],
                    "offsets": [(sectime.get("@start",""), sectime.get("@end",""))],
                    }
                discharge.append(dis)

        print(discharge)
        print("############################################################")



test_3_()

##############################################################################################################################################################################################
"""
        if self.config.schema == "source":
            features = Features(
                {
                    "doc_id": Value("string"),
                    "text": Value("string"),
                    "entities":{
                        "EVENT": Sequence({"id": Value("string"),
                                        "start": Value("int64"),
                                        "end": Value("int64"),
                                        "text": Value("string"),
                                        "modality": ClassLabel(names=["FACTUAL", "CONDITIONAL","POSSIBLE","PROPOSED"]),
                                        "polarity": ClassLabel(names=["POS", "NEG"]),
                                        "type": ClassLabel(names=["TEST","PROBLEM","TREATMENT","CLINICAL_DEPT","EVIDENTIAL","OCCURRENCE"]),
                                        }),
                        "TIMEX3": Sequence({"id": Value("string"),
                                        "start": Value("int64"),
                                        "end": Value("int64"),
                                        "text": Value("string"),
                                        "type": ClassLabel(names=["DATE", "TIME","DURATION","FREQUENCY"]),
                                        "val": Value("string"),
                                        "mod": ClassLabel(names=["NA","MORE","LESS","APPROX","START","END","MIDDLE"]), 
                                        }),
                        "TLINK": Sequence({"id": Value("string"),
                                        "fromID": Value("string"),
                                        "fromText": Value("string"),
                                        "toID": Value("string"),
                                        "toText": Value("string"),
                                        "type": ClassLabel(names=["BEFORE","AFTER","SIMULTANEOUS","OVERLAP","BEGUN_BY","DURING","BEFORE_OVERLAP"]),
                                        }),
                        "SECTIME": Sequence({"id": Value("string"),
                                        "start": Value("int64"),
                                        "end": Value("int64"),
                                        "text": Value("string"),
                                        "type": ClassLabel(names=["ADMISSION","DISCHARGE"]),
                                        "dvalue": Value("string"),                                        
                                        }),
                                 }
                }
            )

"""