import os

import pandas as pd

from config import global_output_directory_name


def merge_candidates_and_abstracts_for_all_keywords():
    all_keywords = os.listdir(global_output_directory_name)
    print(len(all_keywords))
    datasets_to_include = all_keywords
    all_abstracts = pd.DataFrame()
    all_candidates = pd.DataFrame()
    for dataset in datasets_to_include:

        abstracts_file_name = os.path.join(global_output_directory_name, dataset, 'abstracts.csv')
        if os.path.isfile(abstracts_file_name):
            abstracts = pd.read_csv(abstracts_file_name, index_col=[0])
            all_abstracts = all_abstracts.append(abstracts)

        candidates_file_name = os.path.join(global_output_directory_name, dataset, 'relation_candidates.csv')
        if os.path.isfile(candidates_file_name):
            candidates = pd.read_csv(candidates_file_name, index_col=[0])
            all_candidates = all_candidates.append(candidates)

    all_abstracts.drop_duplicates().to_csv(f'{global_output_directory_name}/all_abstracts.csv')
    all_candidates.drop_duplicates().to_csv(f'{global_output_directory_name}/candidates.csv')

def print_stats_for_initial_phase():
    stratified_1000_candidates = pd.read_csv(f'../{global_output_directory_name}/stratified_1000/relation_candidates.csv')
    stratified_1000_df = pd.read_csv(f'../{global_output_directory_name}/stratified_1000/abstracts.txt')
    food_column, disease_column = 'text_x', 'text_y'
    print(stratified_1000_candidates['file_name'].drop_duplicates().shape)

    stratified_1000_candidates = stratified_1000_candidates[
            ['entity_id_y', food_column, disease_column, 'foodon', 'hansard', 'hansardClosest', 'hansardParent', 'snomedct']]
    stratified_1000_candidates = stratified_1000_candidates
    print(stratified_1000_df['PMID'].drop_duplicates().shape)
    print(stratified_1000_df['journal'].drop_duplicates().shape)
    print(stratified_1000_candidates[['text_x','text_y']].shape)

if __name__ == '__main__':
    output_dir= '../results'
    relation_candidates = pd.read_csv(f'{output_dir}/all_candidates.csv', index_col=[0])
    print(f'relation candidates: {relation_candidates.shape[0]}')
    relation_candidates = pd.read_csv(f'{output_dir}/bert_all_candidates.csv', index_col=[0])
    print(f'relation candidates: {relation_candidates.shape[0]}')

    use_existing_df_concatenations=True
    if not use_existing_df_concatenations:
        merge_candidates_and_abstracts_for_all_keywords()

    all_abstracts = pd.read_csv(f'{output_dir}/all_abstracts.csv', index_col=[0])
    print(all_abstracts['journal'].dropna().drop_duplicates().shape[0])

    all_candidates= pd.read_csv(f'{output_dir}/bert_all_candidates.csv', index_col=[0])
    #print(f"processed papers: {all_abstracts['PMID'].drop_duplicates().shape}")
    number_of_unique_pairs = all_candidates[['term1','term2']].drop_duplicates().shape[0]
    print(all_candidates.shape[0])
    print(f"unique pairs: {number_of_unique_pairs}")
    print(f"unique foods: {all_candidates['term1'].drop_duplicates().shape[0]}")
    print(f"unique diseases: {all_candidates['term2'].drop_duplicates().shape[0]}")
    print(f"papers with candidates: {all_candidates['file_name'].drop_duplicates().shape[0]}")

    all_candidates['can_be_linked_to_foodb']=all_candidates['supported_by'].apply(lambda x: x.find('foodb')>-1)
    for food_onto in ['hansardClosest','hansardParent','hansard','snomedct','foodon','foodb_public_id','itis_id','wikipedia_id','ncbi_taxonomy_id']:
        print(f'unique mentions of foods linked to: {food_onto}: {all_candidates[food_onto].dropna().shape[0]}')
        print(f"unique foods linked to {food_onto}: {all_candidates[food_onto].dropna().drop_duplicates().shape[0]}")
        print(f"unique DO-{food_onto} pairs: {all_candidates[[food_onto,'entity_id_y']].dropna().drop_duplicates().shape[0]}")
        print('----------------------')
    print(f"unique mentions of diseases linked to DO: {all_candidates['entity_id_y'].dropna().shape[0]}")
    print(f"unique diseases linked to DO: {all_candidates['entity_id_y'].dropna().drop_duplicates().shape[0]}")