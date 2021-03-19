import pandas as pd
from simpletransformers.classification import ClassificationModel
import os
from relation_classification.dataset_processors import CauseTreatProcessor
from relation_classification.sentence_processors import WordContextWindowRelation
from relation_classification_old.utils import read_df_from_project_dir, save_df_in_project_dir

def apply_relation_classification_on_dataset(dataset, relation_extractor, sentences_in_ground_string = 10, number_of_ground_strings = 10):

    models = {
        'cause': {'files': [{'file_name': 'cause.csv', 'processor': CauseTreatProcessor()}
                            # {'file_name': 'semeval.csv', 'processor': SemEvalProcessor('Cause-Effect(e1,e2)')},
                            # {'file_name': 'semeval.csv', 'processor': SemEvalProcessor('Cause-Effect(e2,e1)')},
                            # {'file_name': 'ade.csv', 'processor':  ADEProcessor('cause')}
                            ],
                  'trained_model': None, 'ground_truth': '', 'entities': ['food', 'disease']},
        'treat': {'files': [{'file_name': 'treat.csv', 'processor': CauseTreatProcessor()}],
                  'trained_model': None, 'ground_truth': '', 'entities': ['food', 'disease']},
    }
    ''''cause_augmented': {'files': [{'file_name': 'cause_augmented.csv', 'processor': AugmentedCauseProcessor()}],
                            'trained_model': None, 'ground_truth': '', 'entities': ['food', 'disease']}'''


    df_t = read_df_from_project_dir(f'{dataset}.csv', 'sentence_relevance_filter')[['text_x', 'text_y', 'sentence_x', 'entity_id_y']]
    df_t = df_t.drop_duplicates()
    df_t['relation_candidates'] = df_t['sentence_x']
    for index, row in df_t.iterrows():
        if len(row['text_x']) > len(row['text_y']):
            df_t.loc[index, 'relation_candidates'] = row['relation_candidates'].replace(row['text_x'], 'XXX').replace(
                row['text_y'], 'YYY')
        else:
            df_t.loc[index, 'relation_candidates'] = row['relation_candidates'].replace(row['text_y'], 'YYY').replace(
                row['text_x'], 'XXX')

    df_t = relation_extractor.extract_relation(df_t)

    models_to_apply = models.keys()
    for model_to_apply in models_to_apply:
        txset = []
        model_name = f'{model_to_apply}_{relation_extractor.name}_{number_of_ground_strings}_{sentences_in_ground_string}'
        model_location = os.path.join('trained_models',model_name)
        ground_strings = read_df_from_project_dir('ground_truth.csv', model_location )
        for index, row in df_t.iterrows():
            for gt in ground_strings['sentence']:
                txset.append([row['relation_candidates'], gt])
        model = ClassificationModel("bert", model_location, use_cuda=False)
        tpredictions, traw_outputs = model.predict(txset)

        txy = pd.DataFrame(txset, columns=['sentence', 'ground_truth'])
        txy['predictions'] = tpredictions
        aggregated_df = txy.groupby('sentence').sum()
        print(aggregated_df)
        aggregated_df[model_to_apply] = [p / number_of_ground_strings for p in aggregated_df['predictions']]
        aggregated_df = aggregated_df[model_to_apply]
        df_t = df_t.merge(aggregated_df, left_on='relation_candidates', right_on=aggregated_df.index)

        pd.set_option('display.expand_frame_repr', False)
        print(df_t.columns)
        print(df_t.shape)
    save_df_in_project_dir(
    df_t[list(models_to_apply) + ['text_x', 'text_y', 'relation_candidates', 'sentence_x', 'entity_id_y']],
    f'{number_of_ground_strings}_{sentences_in_ground_string}_{dataset}.csv',
    'relation_classifier_results')
    return df_t

if __name__ == '__main__':
    relation_extractor = WordContextWindowRelation(5)
    apply_relation_classification_on_dataset('stratified_1000_positive_predictions', relation_extractor)