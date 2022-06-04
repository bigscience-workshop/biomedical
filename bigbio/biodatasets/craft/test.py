import xml.etree.ElementTree as ET

file_path = r"C:\Users\Shamik Bose\Desktop\CRAFT-5.0.0\concept-annotation\GO_BP\GO_BP\knowtator\17565376.txt.knowtator.xml"
tree = ET.parse(file_path)
root = tree.getroot()
for ann in root.findall("annotation"):
    id = ann.find("mention").attrib["id"]
    span_count = ann.findall("span")
    print(len(span_count))
    if len(span_count) > 1:
        print(f"Multiple annotations found for {id}. Skipping")
        continue
    else:
        span = ann.find("span")
        start, end = span.attrib["start"], span.attrib["end"]
        text = ann.find("spannedText").text
        print(id, start, end, text)
