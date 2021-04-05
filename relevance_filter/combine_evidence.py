import os

from config import global_output_directory_name
from utils import read_from_dataset_dir, write_to_dataset_dir, save_as_latex_table
import pandas as pd

def append_evidence_for_relation(relation_df, evidence_df, relation):
    for (term1, term2) in relation_df.index:
        evidence_for_relation = evidence_df[(evidence_df['term1']==term1) & (evidence_df['term2']==term2) & (evidence_df[f'is_{relation}']==1)]
        relation_df.loc[(term1, term2), 'evidence'] = '___'.join(evidence_for_relation['sentence'].values)
    return relation_df


def combine_evidence(dataset,
                     min_positive_classifier_support,
                     max_negative_classifier_support,
                     min_positive_evidence,
                     max_negative_evidence, save_linked=True):

    all_candidates = read_from_dataset_dir('extractors_applied.csv', dataset)
    all_candidates=all_candidates.drop_duplicates()
    cause_columns = list(filter(lambda x: x.find('bert') > -1 and x.find('cause') > -1, all_candidates.columns))
    treat_columns = list(filter(lambda x: x.find('bert') > -1 and x.find('treat') > -1, all_candidates.columns))
    all_candidates['cause_sum'] = all_candidates[cause_columns].sum(axis=1)
    all_candidates['treat_sum'] = all_candidates[treat_columns].sum(axis=1)

    all_candidates['is_cause'] = all_candidates.apply(lambda row: 1 if (row['cause_sum'] >= min_positive_classifier_support and row['treat_sum'] <= max_negative_classifier_support )else 0, axis=1)
    all_candidates['is_treat'] = all_candidates.apply(
        lambda row: 1 if row['treat_sum'] >= min_positive_classifier_support and row['cause_sum'] <= max_negative_classifier_support else 0, axis=1)

    evidence_for_cause_count = all_candidates[all_candidates["is_cause"] == 1].shape[0]
    evidence_for_treat_count = all_candidates[all_candidates["is_treat"] == 1].shape[0]

    print(f'evidence sentences for cause: {evidence_for_cause_count}')
    print(f'evidence sentences for treat: {evidence_for_treat_count}')

    relations = all_candidates.groupby(['term1', 'term2']).sum()
    cause_relations = relations[(relations['is_treat'] <= max_negative_evidence) & (relations['is_cause'] > min_positive_evidence)]
    treat_relations = relations[(relations['is_cause'] <= max_negative_evidence) & (relations['is_treat'] > min_positive_evidence)]

    cause_relations = append_evidence_for_relation(cause_relations, all_candidates, 'cause')
    treat_relations = append_evidence_for_relation(treat_relations, all_candidates, 'treat')
    write_to_dataset_dir(cause_relations, 'cause_relations.csv',dataset)
    write_to_dataset_dir(treat_relations, 'treat_relations.csv', dataset)

    if save_linked:
        get_linked_relations(dataset, all_candidates)


def get_linked_relations(dataset, all_candidates):
    output_dir = os.path.join(global_output_directory_name, dataset, 'linked_relations' )
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    cause_df, treat_df = pd.DataFrame(), pd.DataFrame()
    mappings = {
        'foodon': 'FoodOn', 'snomedct': 'SNOMED CT', 'hansardClosest': 'Hansard Closest',
        'hansardParent': 'Hansard Parent',
        'foodb_public_id': 'FooDB', 'itis_id': 'ITIS', 'wikipedia_id': 'Wikipedia', 'ncbi_taxonomy_id': 'NCBI taxonomy',
        'entity_id_y': 'DO', 'snomedct_disease': 'SNOMED CT', 'umls_disease': 'UMLS', 'nci_disease': 'NCIt',
        'omim_disease': 'OMIM',
        'efo_disease': 'EFO', 'mesh_disease': 'MESH'}
    for food_resource in ['foodon', 'snomedct', 'hansardClosest', 'hansardParent', 'foodb_public_id', 'itis_id',
                          'wikipedia_id', 'ncbi_taxonomy_id']:
        for disease_resource in ['entity_id_y', 'snomedct_disease', 'umls_disease', 'nci_disease', 'omim_disease',
                                 'efo_disease', 'mesh_disease']:
            print(all_candidates.columns)
            relations = all_candidates[
                ['term1', 'term2', food_resource, disease_resource, 'is_cause', 'is_treat']].dropna().groupby(
                ['term1', 'term2', food_resource, disease_resource]).sum()
            cause_relations = relations[(relations['is_treat'] == 0) & (relations['is_cause'] > 1)]
            treat_relations = relations[(relations['is_cause'] == 0) & (relations['is_treat'] > 1)]
            cause_df.loc[mappings[food_resource], mappings[disease_resource]] = int(cause_relations.shape[0])
            treat_df.loc[mappings[food_resource], mappings[disease_resource]] = int(treat_relations.shape[0])
            print(food_resource)
            print(disease_resource)
            print(cause_relations.shape)
            print(treat_relations.shape)
            print('-------------------------')
            cause_relations['relation'] = 'cause'
            treat_relations['relation'] = 'treat'
            write_to_dataset_dir(cause_relations, f'linked_relations/{food_resource}_{disease_resource}_cause.csv', dataset)
            write_to_dataset_dir(treat_relations, f'linked_relations/{food_resource}_{disease_resource}_treat.csv', dataset)
    print(cause_df)
    print(treat_df)

    save_as_latex_table(cause_df, f'{output_dir}/linked_cause_food_disease_final')
    save_as_latex_table(treat_df, f'{output_dir}/linked_treat_food_disease_final')

