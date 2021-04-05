import pandas as pd
import requests
import re

from utils import write_to_dataset_dir, read_from_dataset_dir
import numpy as np
knowledge_base_keys = {'mesh_disease': 'MESH:', 'snomedct_disease':'SNOMEDCT', 'umls_disease':'UMLS_CUI:',
                       'nci_disease': 'NCI:', 'omim_disease': 'OMIM:', 'efo_disease': 'EFO:'}


def get_mappings_from_doid(doid):
    doid_link=f'https://www.ebi.ac.uk/ols/ontologies/doid/terms?iri=http%3A%2F%2Fpurl.obolibrary.org%2Fobo%2F{doid}'
    response=requests.get(doid_link).text
    #print(response)
    found = {}
    for (disease_onto, onto_mention_in_response) in knowledge_base_keys.values():
        pattern_match = re.findall(f'<span>{onto_mention_in_response}\S*</span>', response)
        if len(pattern_match) > 0:
            id = pattern_match[0].replace('<span>','').replace('</span>','')
            found[disease_onto] = id
    return found

def link_do(dataset):
    df =  read_from_dataset_dir('extractors_applied.csv', dataset)
    doids_to_process = df['entity_id_y'].dropna().drop_duplicates().values
    print(doids_to_process.shape)
    for kb in knowledge_base_keys.keys():
        df[kb] = np.nan
    for i in range(0, len(doids_to_process)):
        print(f'{i}/{len(doids_to_process)}')
        doid = doids_to_process[i]
        links_to_other_doids = get_mappings_from_doid(doid.replace(':', '_'))
        other_pairs_with_this_doid = df[df['entity_id_y'] == doid]
        for ontology_name in links_to_other_doids.keys():
            df.loc[other_pairs_with_this_doid.index, ontology_name] = links_to_other_doids[ontology_name]
    write_to_dataset_dir(df, 'extractors_applied.csv', dataset)




def link_foodb(dataset):
    all_candidates = read_from_dataset_dir('extractors_applied.csv', dataset)

    foodb = pd.read_csv('foodb/Food.csv', index_col=[0])
    foodb = foodb.rename({'public_id': 'foodb_public_id'}, axis=1)

    foodb_scientific = foodb.copy()
    foodb_scientific = foodb_scientific[~foodb_scientific['name_scientific'].isna()]
    foodb_scientific['name_scientific'] = foodb_scientific['name_scientific'].apply(lambda x: x.lower())
    foodb_scientific = foodb_scientific.set_index('name_scientific')[
        ['foodb_public_id', 'itis_id', 'wikipedia_id', 'ncbi_taxonomy_id']]

    foodb_non_scientific = foodb.copy()
    foodb_non_scientific = foodb_non_scientific[~foodb_non_scientific['name'].isna()]
    foodb_non_scientific['name'] = foodb_non_scientific['name'].apply(lambda x: x.lower())
    foodb_non_scientific = foodb_non_scientific.set_index('name')[
        ['foodb_public_id', 'itis_id', 'wikipedia_id', 'ncbi_taxonomy_id']]

    foodb = foodb_non_scientific.append(foodb_scientific)
    all_candidates = all_candidates.merge(foodb, left_on='term1', right_on=foodb.index, how='left')

    write_to_dataset_dir(all_candidates, 'extractors_applied.csv', dataset)
