import os
import sys
import traceback
from typing import List

import pandas as pd
import spacy
# from neuralcoref_local import neuralcoref
from spacy.language import Language

from config import global_output_directory_name
from extractors.extractor_base_classes import EntityExtractor, RelationExtractor
from utils import FileUtil


class FooDisExtractor:
    food_extractors: List[EntityExtractor]
    disease_extractors: List[EntityExtractor]

    english_model: Language
    name: str

    def __init__(self, food_extractors: List[EntityExtractor], disease_extractors: List[EntityExtractor], dataset: str = '', verbose=False):
        self.food_extractors = food_extractors
        self.disease_extractors = disease_extractors

        self.english_model = spacy.load('en_core_web_sm')
        self.dataset = dataset
        self.verbose = verbose
        food_extractor_names = '_'.join([extractor.name for extractor in self.food_extractors])
        disease_extractor_names = '_'.join([extractor.name for extractor in self.disease_extractors])
        self.name = 'SABER' + '__'.join([ food_extractor_names, disease_extractor_names])

        # neuralcoref.add_to_pipe(self.english_model, greedyness=0.5)

    def get_food_disease_entities(self, text, file_name=None):
        doc = self.english_model(text)
        disease_noun_chunks = [extractor.extract(doc, file_name, self.dataset) for extractor in self.disease_extractors]
        disease_noun_chunks = [item for sublist in disease_noun_chunks for item in sublist]
        food_noun_chunks = [extractor.extract(doc, file_name, self.dataset) for extractor in self.food_extractors]
        food_noun_chunks = [item for sublist in food_noun_chunks for item in sublist]
        return doc, food_noun_chunks, disease_noun_chunks

    def get_food_disease_dfs(self, text, file_name, dataset):
        doc = self.english_model(text)
        disease_df=None
        disease_df = self.use_extractors(self.disease_extractors, doc, file_name)
        food_df = self.use_extractors(self.food_extractors, doc, file_name)
        return doc, food_df, disease_df

    def use_extractors(self, extractors, doc, file_name):
        df = pd.DataFrame()
        for extractor in extractors:
            extracted_df = extractor.extract(doc, file_name, self.dataset, save_entities=False)
            extracted_df['extractor'] = extractor.name
            df = df.append(extracted_df)
        df['file_name'] = file_name
        return df

    def find_and_save_food_disease_dfs(self, ids_and_abstracts, dataset):

        save_directory = os.path.join(global_output_directory_name, dataset)
        FileUtil.create_directory_if_not_exists(save_directory)

        for extractor in self.food_extractors + self.disease_extractors:
            print(extractor.name)
            df_to_save = pd.DataFrame()
            i = 0
            save_file=os.path.join(save_directory, '{extractor_name}.csv'.format(extractor_name=extractor.name))
            if not os.path.isfile(save_file):
                for (file_name, file_content) in ids_and_abstracts:
                    doc = self.english_model(file_content)
                    #print(i)
                    i += 1
                    file_name = str(file_name)
                    try:
                        extracted_df = extractor.extract(doc, file_name, self.dataset, save_entities=False)
                        extracted_df['extractor'] = extractor.name
                        extracted_df['file_name'] = file_name
                        df_to_save=df_to_save.append(extracted_df)
                    except:
                        if self.verbose:
                            print('Error happened')
                            traceback.print_exc(file=sys.stdout)

                    if i%1000==0:
                        df_to_save.drop_duplicates().to_csv(
                            os.path.join(save_directory, '{extractor_name}_{i}.csv'.format(extractor_name=extractor.name, i = i)))
                if df_to_save.shape[0]==0:
                    df_to_save=pd.DataFrame(columns=['start_char','end_char','entity_type','entity_id','text','sentence','sentence_index','extractor','file_name'])
                df_to_save.drop_duplicates().to_csv(save_file)
            else:
                print('File already exists: {0}'.format(save_file))


