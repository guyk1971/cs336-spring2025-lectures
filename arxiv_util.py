import re
import os
import json
import xml.etree.ElementTree as ET
from arxiv import Client, Search
from dataclasses import asdict
from reference import Reference
from file_util import cached, ensure_directory_exists


def canonicalize(text: str):
    """Remove newlines and extra whitespace with one space."""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def is_arxiv_link(url: str) -> bool:
    return url.startswith("https://arxiv.org/")

def arxiv_reference(url: str, **kwargs) -> Reference:
    """
    Parse an arXiv reference from a URL (e.g., https://arxiv.org/abs/2005.14165).
    Cache the result.
    """
    # Figure out the paper ID
    paper_id = None
    m = re.search(r'arxiv.org\/...\/(\d+\.\d+)(v\d)?(\.pdf)?$', url)
    if not m:
        raise ValueError(f"Cannot handle this URL: {url}")
    paper_id = m.group(1)

    metadata_url = f"http://export.arxiv.org/api/query?id_list={paper_id}"
    metadata_path = cached(metadata_url, "arxiv")
    with open(metadata_path, "r") as f:
        contents = f.read()
    root = ET.fromstring(contents)

    # Extract the relevant metadata
    entry = root.find('{http://www.w3.org/2005/Atom}entry')
    title = canonicalize(entry.find('{http://www.w3.org/2005/Atom}title').text)
    authors = [canonicalize(author.find('{http://www.w3.org/2005/Atom}name').text) for author in entry.findall('{http://www.w3.org/2005/Atom}author')]
    summary = canonicalize(entry.find('{http://www.w3.org/2005/Atom}summary').text)
    published = entry.find('{http://www.w3.org/2005/Atom}published').text

    return Reference(
        title=title,
        authors=authors,
        url=url,
        date=published,
        description=summary,
        **kwargs,
    )



def arxiv_reference_old(url: str, **kwargs) -> Reference:
    """
    Parse an arXiv reference from a URL (e.g., https://arxiv.org/abs/2005.14165).
    Cache the result.
    """
    # Extract arxiv ID from URL
    arxiv_id = url.split("/")[-1].replace(".pdf", "")

    # Read from cache
    ensure_directory_exists("var/arxiv")
    cache_path = os.path.join("var/arxiv", f"{arxiv_id}.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            args = json.load(f)
            return Reference(**json.load(f))
    
    # Query arxiv API
    client = Client()
    print(f"Querying arxiv API for {arxiv_id}")
    search = Search(id_list=[arxiv_id])
    results = list(search.results())
    
    if len(results) != 1:
        raise ValueError(f"Found {len(results)} arXiv papers with ID {arxiv_id}, expected 1")

    # Convert to Reference
    paper = results[0]
    reference = Reference(
        title=paper.title,
        authors=[str(author) for author in paper.authors],
        date=paper.published.strftime("%Y-%m-%d"),
        url=url,
        description=canonicalize(paper.summary),
    )

    # Cache the result
    with open(cache_path, "w") as f:
        json.dump(asdict(reference), f)

    return reference