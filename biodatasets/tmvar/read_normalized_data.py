files = [
    "C:\\Users\\Shamik Bose\\Downloads\\tmVarCorpus\\tmVar.Normalization.txt",
    "C:\\Users\\Shamik Bose\\Downloads\\tmVarCorpus\\test.PubTator.txt",
    "C:\\Users\\Shamik Bose\\Downloads\\tmVarCorpus\\train.PubTator.txt",
]


def generate_raw_docs(fstream):
    raw_document = []
    for line in fstream:
        if line.strip():
            raw_document.append(line.strip())
        elif raw_document:
            yield raw_document
            raw_document = []
    if raw_document:
        yield raw_document


def parse_raw_doc(raw_doc):
    pmid, _, title = raw_doc[0].split("|")
    pmid = int(pmid)
    _, _, abstract = raw_doc[1].split("|")
    passages = [
        {"type": "title", "text": [title], "offsets": [[0, len(title)]]},
        {
            "type": "abstract",
            "text": [abstract],
            "offsets": [[len(title) + 1, len(title) + len(abstract) + 1]],
        },
    ]
    entities = []
    for line in raw_doc[2:]:
        mentions = line.split("\t")
        # Hacky fix for inconsistent formats between v1 and v2
        if len(mentions) == 6:
            (pmid_, start_idx, end_idx, mention, semantic_type_id, entity_id) = mentions
            rsid = None
        elif len(mentions) == 7:
            (
                pmid_,
                start_idx,
                end_idx,
                mention,
                semantic_type_id,
                entity_id,
                rsid,
            ) = mentions

        entity = {
            "offsets": [[int(start_idx), int(end_idx)]],
            "text": [mention],
            "semantic_type_id": semantic_type_id.split(","),
            "concept_id": entity_id,
            "rsid": rsid,
        }
        entities.append(entity)
    return {"pmid": pmid, "passages": passages, "entities": entities}


for file in files:
    print(file)
    docs = generate_raw_docs(open(file, "r"))
    for doc in docs:
        parse_raw_doc(doc)

""C:\Users\Shamik Bose\Downloads\tmVarCorpus\test.PubTator.txt""