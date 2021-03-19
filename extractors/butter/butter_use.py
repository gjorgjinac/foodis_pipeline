import os
import re

from nltk import word_tokenize

from extractors.butter.BILSTM_CRF_model import BILSTMCRFModel
from extractors.butter.model_definition import models

os.environ["HDF5_DISABLE_VERSION_CHECK"] = "1"
from BILSTM_CharEmb_model import BILSTMDoubleInputModel
from utils import get_char_indices, \
    SentenceGetter, get_label
import numpy as np
import tensorflow as tf
import pandas as pd

# --------------SETTINGS-------------

max_sentence_length = 50
EPOCHS = 1000
BATCH_SIZE = 256
EMBEDDING = 300
pre_proc = "lemma"
vectorizer_model_name = 'lexical'
missing_values_handled = False
task_name = "food-classification"
include_char_embeddings = False
# --------------SETTINGS-------------

#full_data, train_data, test_data = read_data_for_task(task_name)

model_instance = BILSTMDoubleInputModel() if include_char_embeddings else BILSTMCRFModel()

word2idx = pd.read_csv('word2idx', index_col=[0])
word2idx = {t[0]: t[1] for t in list(word2idx.values)}

char2idx = pd.read_csv('char2idx', index_col=[0])
char2idx = {t[0]: t[1] for t in list(char2idx.values)}

idx2tag = pd.read_csv('idx2tag', index_col=[0])
idx2tag = {t[0]: t[1] for t in list(idx2tag.values)}

full_model_name = f"{vectorizer_model_name}_{pre_proc}_{missing_values_handled}_e{EPOCHS}_emb{models[vectorizer_model_name]['vector_size']}.h5"

max_word_length = np.max([len(str(w)) for w in word2idx.keys()])

model = tf.keras.models.load_model(full_model_name)
model.summary()
file_directory= '../../sample_texts/inputs/'
for file_name in os.listdir(file_directory):
    file_path = f'{file_directory}/{file_name}'
    file = open(file_path, mode='r')
    test_file_content = file.read()
    file.close()
    #print(test_file_content)
    #test_file_content = re.sub(r"[^\w\s.\d]", "", test_file_content)

    test_words = word_tokenize(test_file_content)
    test_data = pd.DataFrame(test_words, index=test_words)
    X_te = model_instance.process_X(test_data, word2idx, max_sentence_length)
    X_char_te = get_char_indices(test_data, max_word_length, max_sentence_length, char2idx)
    sentence_getter = SentenceGetter(test_data, label_adapter=get_label)
    padding_start = [len(s) for s in sentence_getter.sentences]
    X_te = np.asarray(X_te).astype(np.float32)

    sentence_preds = model.predict([X_te, X_char_te])
    sentence_preds = [s[:padding_start_index] for s, padding_start_index in zip(sentence_preds, padding_start)]

    predicted_labels = [idx2tag[np.argmax(prob)] for sentence in sentence_preds for prob in sentence]
    food_entities = []

    all_sentences = list(test_data.index)
    food_pieces = []
    for index, label in enumerate(predicted_labels):
        if label == 'B-FOOD':
            food_pieces=[all_sentences[index]]
        elif label == 'I-FOOD' and len(food_pieces) > 0:
            food_pieces.append(all_sentences[index])
        else:
            if len(food_pieces) > 0:
                food_entities.append(' '.join(food_pieces))
                food_pieces = []
    print(food_entities)
