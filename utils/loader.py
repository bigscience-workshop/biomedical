import json
import os
import time
from typing import List

from Bio import Entrez
from tqdm import tqdm


def fetch_pubmed_abstracts(
    pmids: List,
    outdir: str,
    cred_mail: str,
    batch_size: int = 100,
    delay: float = 0.3,
    overwrite: bool = False,
    verbose: bool = True,
):
    """
    Fetches pubmed articles for a given list of PMIDs.

    PubMed articles can be downloaded in bulks, for now tested with up to 1000 articles per requests,
    but can still be slow. The BioASQ Task C 2017 contains up to 63 000 articles per split.

    Also required is a email address which is registered at https://pubmed.ncbi.nlm.nih.gov,
    we therefore discussed a extra attribute "cred_mail" in the BigBioConfig class, that can
    optionally be filled out. Additionally the dependecy `biopython` has to be introduced
    (https://biopython.org).

    For now this function also dumps all articles in arbitrary named json files in outdir.

    """

    Entrez.email = cred_mail

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    if not isinstance(pmids, list):
        pmids = list(pmids)

    n_chunks = int(len(pmids) / batch_size)

    for i in tqdm(range(0, n_chunks + (1 if n_chunks * batch_size < len(pmids) else 0))):
        start, end = i * batch_size, (i + 1) * batch_size
        outfname = f"{outdir}/pubmed.{i}.json"
        if os.path.exists(outfname) and not overwrite:
            continue

        query = ",".join(pmids[start:end])
        handle = Entrez.efetch(db="pubmed", id=query, rettype="gb", retmode="xml", retmax=batch_size)
        record = Entrez.read(handle)
        if len(record["PubmedArticle"]) != len(pmids[start:end]) and verbose:
            print(f"Queried {len(pmids[start:end])}, returned {len(record['PubmedArticle'])}")

        time.sleep(delay)
        # dump to JSON
        with open(outfname, "wt") as file:
            file.write(json.dumps(record, indent=2))
