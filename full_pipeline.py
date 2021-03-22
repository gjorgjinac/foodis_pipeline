import os

import pandas as pd

from abstract_download.download_abstracts import download_and_save_abstracts_for_search_term
from abstract_download.pubmed_util import append_to_processed
from config import global_output_directory_name
from relevance_filter.combine_evidence import combine_evidence
from extractors.butter_food_extractor import ButterFoodExtractor
from extractors.foodb_food_extractor import FoodbFoodExtractor
from extractors.foodis_extractor import FooDisExtractor
from extractors.foodner_web_food_extractor import FoodnerWebFoodExtractor
from extractors.saber_biomed_extractor import SaberBioMedExtractor
from relevance_filter.generate_candidates_from_dfs import get_relation_candidates_from_datasets
from food_linkage.join_food import join_food_extractions
from food_linkage.link_do_foodon import link_foodb, link_do
from relevance_filter.bert_simple_apply import apply_relation_classification_bert_simple
from relevance_filter.sentence_relevance_apply import apply_sentence_relevance_filter
from utils import add_span_extensions


def get_foodis_extractor():
    return FooDisExtractor([
        FoodnerWebFoodExtractor(),
        FoodbFoodExtractor(save_extractions=False, include_scientific=False, include_non_scientific=True),
        FoodbFoodExtractor(save_extractions=False, include_scientific=True, include_non_scientific=False),
        ButterFoodExtractor(path_to_module='./extractors/butter', epochs=1000, save_extractions=False)
    ],[SaberBioMedExtractor(save_extractions=False)], '', verbose=False)


def append_to_existing(candidates):
    all_candidates_file = f'{global_output_directory_name}/all_candidates.csv'
    if os.path.isfile(all_candidates_file):
        df = pd.read_csv(all_candidates_file, index_col=[0])
        candidates = df.append(candidates)
    candidates.to_csv(all_candidates_file)


def run_full_foodis_pipeline(search_terms,
                             number_of_abstracts=10,
                             min_food_support=2,
                             min_positive_classifier_support=3,
                             max_negative_classifier_support=1,
                             min_positive_evidence=1,
                             max_negative_evidence=0,
                             foodis_extractor=None
                             ):
    add_span_extensions()
    search_terms = [search_terms] if type(search_terms) is str else search_terms
    if foodis_extractor is None:
        foodis_extractor = get_foodis_extractor()
    for search_term in search_terms:
        dataset = search_term.replace(' ', '_')
        print(dataset)
        abstracts_df = download_and_save_abstracts_for_search_term(search_term, dataset,
                                                                   max_ids=number_of_abstracts)
        if abstracts_df is not None and abstracts_df.shape[0] > 0:
            ids_and_abstracts = abstracts_df.loc[:, ['PMID', 'abstract']].dropna().values
            foodis_extractor.dataset = dataset
            foodis_extractor.find_and_save_food_disease_dfs(ids_and_abstracts, dataset)
            food_file_name, _ = join_food_extractions(dataset, min_support=min_food_support)
            food_file_name = f'foods_support_{min_food_support}.csv'
            candidates = get_relation_candidates_from_datasets([dataset], food_file_name, 'saber_diso.csv')
            append_to_processed(abstracts_df['PMID'].values)
            append_to_existing(candidates)
            apply_sentence_relevance_filter(dataset, 'relevance_filter')
            apply_relation_classification_bert_simple(dataset, 'relevance_filter')
            combine_evidence(dataset,
                             min_positive_classifier_support=min_positive_classifier_support,
                             max_negative_classifier_support=max_negative_classifier_support,
                             min_positive_evidence=min_positive_evidence,
                             max_negative_evidence=max_negative_evidence)
            link_foodb(dataset)
            link_do(dataset)


if __name__ == '__main__':
    keywords_to_process = ['lemon bronchitis']
    run_full_foodis_pipeline(keywords_to_process, number_of_abstracts=300)
