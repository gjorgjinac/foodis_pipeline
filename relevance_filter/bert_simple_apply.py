import os

import pandas as pd
from simpletransformers.classification import ClassificationModel

from config import use_gpu, global_output_directory_name
from relevance_filter.dataset_processors import CauseTreatProcessor, ADEProcessor
from relevance_filter.sentence_processors import WordContextWindowRelation
from utils import get_dataset_output_dir, read_from_dataset_dir, write_to_dataset_dir


def apply_relation_classification_bert_simple(dataset, path_to_module):
    config = {
        'cs': {'processor': CauseTreatProcessor(), 'tasks': ['cause', 'treat']},
        'ade': {'processor': ADEProcessor('cause'), 'tasks': ['cause']},
        'foodis': {'processor': CauseTreatProcessor(), 'tasks': ['cause', 'treat']},
    }

    bert_models_to_apply = ['biobert', 'roberta']
    sources_to_apply = ['foodis', 'cs']

    relation_extractor = WordContextWindowRelation(5)
    context_extraction_applied = False
    relevant_sentences = read_from_dataset_dir('relevant_candidates.csv', dataset)
    if relevant_sentences.shape[0] > 0:
        if 'sentence' not in relevant_sentences.columns:
            sentence_column = 'sentence_x' if 'sentence_x' in relevant_sentences.columns else 'sentence_y'
            relevant_sentences = relevant_sentences.rename(columns={sentence_column: 'sentence'})
        relevant_sentences = relevant_sentences.rename(columns={'text_x': 'term1', 'text_y': 'term2'})

        for source in sources_to_apply:
            if 'relation_candidates' not in relevant_sentences.columns:
                relevant_sentences['relation_candidates'] = relevant_sentences.apply(
                    lambda row: config[source]['processor'].extract_relation(row), axis=1)
            if not context_extraction_applied:
                relevant_sentences = relation_extractor.extract_relation(relevant_sentences)
                context_extraction_applied = True
            simple_models_to_apply = [(bert_model_name, task_name, source) for bert_model_name in bert_models_to_apply for
                                      task_name in config[source]['tasks']]
            for bert_model_name, task_name, source in simple_models_to_apply:
                trained_model = ClassificationModel(
                    bert_model_name if bert_model_name != 'biobert' else 'bert',
                    f'trained_models/{bert_model_name}_{task_name}_{source}_{relation_extractor.name}/best',
                    use_cuda=use_gpu
                )
                relevant_sentences = relation_extractor.extract_relation(relevant_sentences)
                apply_predictions, apply_raw_outputs = trained_model.predict(
                    relevant_sentences['relation_candidates'].values)
                relevant_sentences[f'{bert_model_name}_{task_name}_{source}_simple'] = apply_predictions

    write_to_dataset_dir(relevant_sentences, 'extractors_applied.csv', dataset)


if __name__ == '__main__':
    dataset = 'stratified_1000'
    df = pd.read_csv(f'{global_output_directory_name}/{dataset}/relevant_candidates.csv', index_col=[0])
    apply_relation_classification_bert_simple(df, dataset)