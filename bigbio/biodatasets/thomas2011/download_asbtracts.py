import requests
import os
import xml.etree.ElementTree as ET

for id in [
    "16338218",
    "15645182",
]:
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
    params = {
        "db": "pubmed",
        "id": id,
        "retmode": "xml",
        "rettype": "medline",
    }
    res = requests.get(url, params=params)
    blank_line_count = 0
    required_text_lines = []
    tree = ET.XML(res.text)
    article = tree.find("PubmedArticle").find("MedlineCitation").find("Article")
    article_title = article.find("ArticleTitle").text
    abstract_parts = [f"{article_title}"]
    article_abstract = article.find("Abstract").findall("AbstractText")
    for abstract_part in article_abstract:
        label = abstract_part.attrib.get("Label", "")
        if label:
            abstract_parts.append(f"{label}:{abstract_part.text}")
        else:
            abstract_parts.append(abstract_part.text)
    print(f'PMID: {id}\n{"".join(abstract_parts)}')
    print("|=" * 45)
