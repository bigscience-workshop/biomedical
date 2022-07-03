import requests
import os
import xml.etree.ElementTree as ET

for id in ["17563728", "17548681", "17566096"]:
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
    abstract_parts = [article_title]
    article_abstract = article.find("Abstract").findall("AbstractText")
    for abstract_part in article_abstract:
        print(abstract_part.attrib)
        abstract_parts.append(abstract_part.text)
    print(
        f'PMID: {id}\nTitle: {abstract_parts[0]}\nAbstract: {"".join(abstract_parts[1:])}'
    )
    print("|=" * 45)
