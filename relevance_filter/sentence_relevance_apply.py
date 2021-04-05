import os

import pandas as pd
from simpletransformers.classification import ClassificationModel

from config import use_gpu
from utils import get_dataset_output_dir, write_to_dataset_dir


def apply_sentence_relevance_filter(dataset, path_to_module=''):
    dataset_output_dir = get_dataset_output_dir(dataset)
    df=pd.read_csv(os.path.join(dataset_output_dir, 'relation_candidates.csv' ), index_col=[0])
    if 'sentence' not in df.columns:
        df = df.rename({'sentence_x': 'sentence'}, axis=1)
    df = df.reset_index(drop=True)
    print(df.index)
    df=df[~df['sentence'].isna()]
    model_location = os.path.join('trained_models', 'sentence_relevance')
    print(df['sentence'].values)
    trained_model = ClassificationModel(
        "bert", model_location, use_cuda=use_gpu
    )
    relevant_df = pd.DataFrame()
    if df.shape[0] > 0:
        apply_predictions, apply_raw_outputs = trained_model.predict(list(df['sentence'].values))
        df['prediction'] = apply_predictions
        df[['sentence', 'prediction']].drop_duplicates().to_csv(os.path.join(dataset_output_dir, f'sentence_relevance_values.csv'))
        df[df['prediction'] == 1].to_csv(os.path.join(dataset_output_dir,f'relevant_candidates.csv'))
        df[df['prediction'] == 0].to_csv(os.path.join(dataset_output_dir,f'irrelevant_candidates.csv'))
        relevant_df= df[df['prediction'] == 1]

    write_to_dataset_dir(relevant_df, 'relevant_candidates.csv', dataset)
    return relevant_df

if __name__ == '__main__':
    df = pd.read_csv(f'all_relation_candidates.csv', index_col=[0]).dropna('sentence')
    apply_sentence_relevance_filter(df, 'stratified_1000_support_3')