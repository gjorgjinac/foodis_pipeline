import os

import requests
from lxml import etree
import pandas as pd
import numpy as np
import math
import time

from config import global_output_directory_name


def append_to_processed(ids):
    processed_ids_file = f'{global_output_directory_name}/processed_ids.csv'
    processed = pd.DataFrame()
    if os.path.isfile(processed_ids_file):
        processed = pd.read_csv(processed_ids_file)['pmid']
    pd.DataFrame(set(processed.values).union(set(ids)), columns=['pmid']).to_csv(processed_ids_file)


def retrieve_pubmed_abstracts(search_terms, max_ids):
    doc_ids = search_pubmed(search_terms)
    print('ids', len(doc_ids))
    processed = {}
    if os.path.isfile(f'{global_output_directory_name}/processed_ids.csv'):
        processed = set(pd.read_csv(f'{global_output_directory_name}/processed_ids.csv')['pmid'].astype('str').values)
    doc_ids = set(doc_ids).difference(processed)
    print('new ids:', len(doc_ids))
    if len(doc_ids) == 0:
        return pd.DataFrame()
    doc_ids = list(doc_ids)[0:max_ids]

    doc_info = get_paper_data(doc_ids)
    return doc_info


def search_pubmed(search_terms):
    url = construct_url(search_terms, 'search')
    xml = fetch(url)
    root = etree.fromstring(xml)
    return [i.text for i in root.findall('.//Id')]


def get_paper_data(paper_ids):
    num_divisions = int(math.ceil(len(paper_ids) / 100))
    split_ids = np.array_split(np.asarray(paper_ids), num_divisions)
    paper_ids = [np.ndarray.tolist(split_ids[i]) for i in range(len(split_ids))]

    paper_data = []
    for i in paper_ids:
        url = construct_url(i, 'document')
        try:
            xml = fetch(url)
            root = etree.fromstring(xml)
            paper_data = paper_data + root.findall('PubmedArticle')
        except Exception:
            print('Exception thrown when getting paper data')
    info = pd.DataFrame()

    for single_paper_data in paper_data:
        mesh_terms = []
        mesh_ids = []
        for mesh_section in single_paper_data.findall('.//MeshHeading'):
            mesh_terms.append(mesh_section.find('.//DescriptorName').text)
            mesh_ids.append(mesh_section.find('.//DescriptorName').attrib['UI'])

        pmid = int(single_paper_data.find('.//PMID').text)
        abstract_node = single_paper_data.find('.//AbstractText')
        paper_row = {
            'PMID': int(pmid),
            'abstract': abstract_node.text if abstract_node is not None else None,
            'paper': single_paper_data.find('.//ArticleTitle').text,
            'journal': single_paper_data.find('.//Title').text,
            'year': single_paper_data.find('.//Year').text,
            'mesh_terms': mesh_terms,
            'mesh_ids': mesh_ids,
            'webpage': 'https://www.ncbi.nlm.nih.gov/pubmed/' + str(pmid)
        }

        info = info.append(paper_row, ignore_index=True)
    return info.reset_index(drop=True)


def construct_url(url, query, num_results=1000000):
    if query == 'search':
        base_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term='
        term_url = '%20AND%20'.join([s.replace(" ", "%20") for s in url])
        max_results = '&retmax=' + str(num_results)
        return base_url + term_url + max_results

    elif query == 'document':
        base_url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id='
        return base_url + ','.join(url) + '&retmode=xml'
    else:
        print('Invalid query type')


def fetch(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.content
        elif response.status_code == 404:
            return None
        else:
            time.sleep(1)
            return fetch(url)
    except TimeoutError:
        time.sleep(2)
        return fetch(url)
