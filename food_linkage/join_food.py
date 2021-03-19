import os

import nltk
import numpy as np
import pandas as pd
from nltk import word_tokenize

from config import global_output_directory_name
from utils import read_from_dataset_dir


def remove_overlapping_foods(df):
    to_remove = []
    df = df.reset_index(drop=True)
    for i, row_i in df.iterrows():
        foods_in_file = df[df['sentence_index'] == row_i['sentence_index']][df['file_name'] == row_i['file_name']]
        for j, row_j in foods_in_file.iterrows():
            if row_i['text'] != row_j['text'] and row_i['text'].find(row_j['text']) > -1 and row_i['start_char'] <= \
                    row_j['start_char'] \
                    and row_i['end_char'] >= row_j['end_char']:
                to_remove.append(j)
                print(row_j['text'])
            if row_i['text'] == row_j['text'] and row_i['start_char']==row_j['start_char']:
                if row_i['support']>row_j['support']:
                    to_remove.append(j)
                if row_i['support'] < row_j['support']:
                    to_remove.append(i)
    print(f'removing {len(to_remove)}')
    df = df.drop(to_remove)
    return df


def merge_food_dfs(df1, df2, df1_name, df2_name):
    if df1.shape[0] ==0:
        return df2
    if df2.shape[0]==0:
        return df1
    df1_suffix = f'_{df1_name}' if df1_name is not None else ''
    df2_suffix = f'_{df2_name}' if df2_name is not None else ''
    df = df1.merge(df2, left_on=['sentence_index', 'file_name'], right_on=['sentence_index', 'file_name'], how='outer',
                   suffixes=[df1_suffix, df2_suffix])
    columns_to_rename = { 'text':None, 'start_char': None, 'end_char': None}
    for column in columns_to_rename.keys():
        columns_to_rename[column] = f'{column}{df2_suffix}'
    df = df.drop_duplicates()
    return df.rename(columns=columns_to_rename)


def read_food_file(dataset, file_name):
    df = pd.read_csv(f'{global_output_directory_name}/{dataset}/{file_name}', index_col=[0])
    if df.shape[0] > 0:
        df['text'] = df['text'].fillna('').apply(lambda x: x.lower())
    print(df.shape)
    return df

def join_food_extractions(dataset, min_support):
    foods_butter, foods_foodb_non_scientific, foods_foodb_scientific, foods_foodner_web \
        = [read_food_file(dataset, file) for file
           in
           ['butter.csv',
            'foodb_scientific_False_non_scientific_True.csv',
            'foodb_scientific_True_non_scientific_False.csv',
            'foodner_web.csv'
            ]]

    extractors = {
                  'foodner_web': foods_foodner_web,
                  'butter': foods_butter,
                  'foodb_non_scientific': foods_foodb_non_scientific}


    foods_merged = merge_food_dfs(foods_foodb_non_scientific, foods_butter,'foodb_non_scientific', 'butter')
    foods_merged = merge_food_dfs(foods_merged, foods_foodner_web, None, 'foodner_web')
    extractor_names=list(extractors.keys())
    extractor_text_columns=set([f'text_{extractor}' for extractor in extractor_names]).intersection(foods_merged.columns)
    foods_filtered = pd.DataFrame()
    for index, row in foods_merged.iterrows():
        extracted_foods = [row[col] for col in extractor_text_columns]
        has_support = {}
        for extractor in extractor_names:
            extracted_text = row[f'text_{extractor}']
            has_support[extractor] = (len(list(filter(lambda x: type(extracted_text) is str and type(x) is str
                                                                and extracted_text.find(x)>-1, extracted_foods))) >= min_support)
        support = len(list(filter(lambda x: x, has_support.values())))
        is_linked_by_foodner = has_support['foodner_web'] and len(list(filter(lambda x: row[x] is not None, ['foodon','hansard','hansardClosest','hansardParent','snomedct'] ))) > 0
        if support >= min_support:
            get_data_from_extractor = list( filter(lambda ext: has_support[ext], ['foodner_web','butter','foodb_non_scientific'])) [0]
            for new_column in {'text', 'start_char', 'end_char', 'sentence'}.difference(row.keys()):
                row[new_column] = row[f'{new_column}_{get_data_from_extractor}']
            row['data_from_extractor'] = get_data_from_extractor
            row['supported_by'] = '|'.join(list(filter(lambda k: has_support[k], extractor_names)))
            row['support']=support
            if row['text'].lower() not in {'food','foods','consumption','drug','drugs'}:
                foods_filtered = foods_filtered.append(row)
    if foods_filtered.shape[0]!=0:
        foods_filtered = foods_filtered.drop([f'text_{extractor}' for extractor in extractor_names], axis=1)
        foods_filtered['extractor'] = '_'.join(extractor_names)

        all_removed_foods = set()
        if foods_filtered.shape[0] > 0:
            foods_filtered['contains_noun'] = foods_filtered.apply(
                lambda row: np.any([nltk.pos_tag([word])[0][1] in ['NN', 'NNS', 'NNP', 'NNPS'] for word in
                                    word_tokenize(str(row['text']))]), axis=1)
            foods = set(foods_filtered['text'])
            foods_filtered = foods_filtered[foods_filtered['contains_noun']]
            are_foods = set(foods_filtered['text'])
            removed_foods = foods.difference(are_foods)
            if len(removed_foods) > 0:
                all_removed_foods = all_removed_foods.union(removed_foods)
                print(all_removed_foods)

        foods_filtered = foods_filtered[['data_from_extractor',
                                         'end_char',
                                         'entity_id',
                                         'entity_type',
                                         'extractor',
                                         'file_name',
                                         'sentence',
                                         'sentence_index',
                                         'start_char',
                                         'text',
                                         'support',
                                         'supported_by',
                                          'foodon','hansard','hansardClosest','hansardParent','snomedct','synonyms'
                                         ]]
        foods_foodb_scientific['extractor']='foodb_scientific'
        foods_foodb_scientific['support']=1
        foods_foodb_scientific['supported_by']='foodb_scientific'
        foods_filtered = foods_filtered.append(foods_foodb_scientific)
        foods_filtered = foods_filtered.drop_duplicates()
        foods_filtered = remove_overlapping_foods(foods_filtered)
    food_file_name = f'foods_support_{min_support}.csv'

    diseases = read_from_dataset_dir('saber_diso.csv', dataset)['text']
    foods_filtered = foods_filtered[~foods_filtered['text'].isin(diseases)]
    foods_filtered.to_csv(os.path.join(global_output_directory_name, dataset, food_file_name))
    return food_file_name, foods_filtered
