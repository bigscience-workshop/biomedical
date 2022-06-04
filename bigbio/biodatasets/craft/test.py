import xml.etree.ElementTree as ET

file_path = r"C:\Users\Shamik Bose\Downloads\CRAFT-5.0.0\CRAFT-5.0.0\concept-annotation\CHEBI\CHEBI\knowtator\11319941.txt.knowtator.xml"
tree = ET.parse(file_path)
root = tree.getroot()
for ann in root.findall("annotation"):
    id = ann.find("mention").attrib["id"]
    span = ann.find("span")
    start, end = span.attrib["start"], span.attrib["end"]
    text = ann.find("spannedText").text
    print(id, start, end, text)
