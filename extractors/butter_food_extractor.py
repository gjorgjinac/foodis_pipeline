from typing import List

import numpy as np
import pandas as pd
from nltk import word_tokenize
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
from tensorflow.keras.models import Model

from extractors.butter.BILSTM_CRF_model import BILSTMCRFModel
from extractors.butter.BILSTM_CharEmb_model import BILSTMDoubleInputModel
from extractors.butter.model_definition import models
from extractors.butter.utils import get_char_indices
from extractors.extractor_base_classes import EntityExtractor


class ButterFoodExtractor(EntityExtractor):
    word2idx: dict
    char2idx: dict
    idx2tag: dict
    max_word_length: int
    max_sentence_length: int
    full_model_name: str
    model_instance: any
    model: Model

    def __init__(self, path_to_module='', vectorizer_model_name='lexical', pre_proc='lemma',
                 missing_values_handled=False, epochs=1000, save_extractions=True, include_char_embeddings=False):
        super().__init__('butter', save_extractions=save_extractions)
        self.initialize_dictionaries(path_to_module)
        self.max_word_length = np.max([len(str(w)) for w in self.word2idx.keys()])
        self.max_sentence_length = 50
        self.full_model_name = f"trained_models/{vectorizer_model_name}_{pre_proc}_{missing_values_handled}_e{epochs}_emb{models[vectorizer_model_name]['vector_size']}.h5"
        self.model_instance = BILSTMDoubleInputModel() if include_char_embeddings else BILSTMCRFModel()
        self.__initialize_model( vectorizer_model_name)

    def __initialize_model(self, vectorizer):
        self.model = self.model_instance.get_compiled_model(vectorizer, False,
                                                            self.max_sentence_length,
                                                            self.max_word_length,
                                                            len(self.word2idx.keys()),
                                                            len(self.char2idx.keys()),
                                                            len(self.idx2tag.keys()),
                                                            self.word2idx
                                                            )

        self.model.load_weights(self.full_model_name)

    def initialize_dictionaries(self, base_directory=''):
        word2idx = pd.read_csv(f'{base_directory}/word2idx', index_col=[0])
        self.word2idx = {t[0]: t[1] for t in list(word2idx.values)}

        char2idx = pd.read_csv(f'{base_directory}/char2idx', index_col=[0])
        self.char2idx = {t[0]: t[1] for t in list(char2idx.values)}

        idx2tag = pd.read_csv(f'{base_directory}/idx2tag', index_col=[0])
        self.idx2tag = {t[0]: t[1] for t in list(idx2tag.values)}

    def extract_entity(self, doc: Doc) -> List[Span]:

        data_as_sentences = [[word.lower() for word in word_tokenize(sentence.text)] for sentence in doc.sents]

        X_te = self.model_instance.process_X(data_as_sentences, self.word2idx, self.max_sentence_length, True)
        X_char_te = get_char_indices(data_as_sentences, self.max_word_length, self.max_sentence_length, self.char2idx,
                                     True)
        padding_start = [len(s) for s in data_as_sentences]
        X_char_te = np.nan_to_num(np.asarray(X_char_te).astype(np.float32))
        X_te = np.asarray(X_te)
        sentence_preds = self.model.predict([X_te, X_char_te])
        sentence_preds = [s[:padding_start_index] for s, padding_start_index in zip(sentence_preds, padding_start)]
        predicted_labels = [self.idx2tag[tag_index] for sentence in sentence_preds for tag_index in sentence]
        food_entities = []

        data_as_list_of_words = [word for sentence in data_as_sentences for word in sentence]
        food_pieces = []
        for index, label in enumerate(predicted_labels):

            if label == 'B-FOOD':
                food_pieces = [data_as_list_of_words[index]]
            elif label == 'I-FOOD' and len(food_pieces) > 0:
                food_pieces.append(data_as_list_of_words[index])
            else:
                if len(food_pieces) > 0:
                    food_entities.append(' '.join(food_pieces))
                    food_pieces = []
        return self.__convert_food_words_to_doc(doc, food_entities)

    def __convert_food_words_to_doc(self, doc, food_words):
        text_lowercase = doc.text.lower()
        food_words = list(filter(lambda food: text_lowercase.find(food) > -1, food_words))
        return [doc.char_span(text_lowercase.find(food), text_lowercase.find(food) + len(food)) for food in food_words]
