import os

import pandas as pd
import spacy

from config import global_output_directory_name
from utils import PandasUtil

def get_output_directory(dataset):
    return os.path.join(global_output_directory_name, dataset)

def get_relation_candidates_from_datasets(datasets, food_file_name, disease_file_name):
    food_df = pd.DataFrame()
    disease_df = pd.DataFrame()
    english_model = spacy.load('en_core_web_sm')
    for dataset in datasets:
        dataset_output_dir=get_output_directory(dataset)
        foods_in_dataset = PandasUtil.read_dataframe_file(food_file_name, dataset_output_dir).fillna('')
        foods_in_dataset['dataset'] = dataset
        food_df = food_df.append(foods_in_dataset)

        diseases_in_dataset = PandasUtil.read_dataframe_file(disease_file_name, dataset_output_dir)
        diseases_in_dataset['dataset']=dataset
        disease_df=disease_df.append(diseases_in_dataset)

    if food_df.shape[0] != 0 and disease_df.shape[0] != 0:
        both_df = food_df.merge(disease_df, left_on=['sentence_index', 'file_name','dataset'], right_on=['sentence_index', 'file_name','dataset'])
        both_df = both_df.drop(both_df[both_df['text_x'] == both_df['text_y']].index)
        extractions_from_all_files = both_df.drop_duplicates()
        #extractions_from_all_files.to_csv(f'relation_classification/relation_candidates/{"_".join(datasets)}.csv')

        if len(datasets)==1:
            extractions_from_all_files.to_csv(os.path.join(global_output_directory_name, datasets[0], 'relation_candidates.csv'))

        return extractions_from_all_files
    else:
        print(f'Food or disease df is empty. Food df shape: {food_df.shape} Disease df shape: {disease_df.shape}')


