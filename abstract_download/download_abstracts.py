import os

from abstract_download.pubmed_util import retrieve_pubmed_abstracts
from config import global_output_directory_name
from utils import FileUtil


def download_and_save_abstracts_for_search_term(search_term, dataset, max_ids):
    abstracts_df = retrieve_pubmed_abstracts([search_term], max_ids)
    dataset_output_directory = os.path.join(global_output_directory_name,dataset)
    FileUtil.create_directory_if_not_exists(dataset_output_directory)
    abstracts_df.to_csv(os.path.join(dataset_output_directory, 'abstracts.csv'))
    return abstracts_df