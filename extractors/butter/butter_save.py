import os

from extractors.butter.BILSTM_CRF_model import BILSTMCRFModel
from extractors.butter.model_definition import models
from extractors.butter.utils import read_index_from_file
from BILSTM_CharEmb_model import BILSTMDoubleInputModel
from utils import save_report_to_file, get_pred_and_ground_string, get_char_indices, \
    get_char_to_index_dict, get_word_to_index_mappings, get_tag_to_index_mappings, \
    read_data_for_task
import numpy as np
import tensorflow as tf
import random
from sklearn.metrics import classification_report
from nltk.stem import WordNetLemmatizer
import nltk
import pandas as pd

# --------------SETTINGS-------------

max_sentence_length = 50
EPOCHS = 10
BATCH_SIZE = 256
EMBEDDING = 300
pre_proc = "none"
vectorizer_model_name = 'lexical'
missing_values_handled = False
task_name = "food-classification"
regenerate_index = False
include_char_embeddings=False

# --------------SETTINGS-------------

# seeds
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

full_data, train_data, test_data = read_data_for_task(task_name, './data')
# train_data = full_data
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

model_instance = BILSTMDoubleInputModel() if include_char_embeddings else BILSTMCRFModel()


def lemmatize_df_index(df):
    df.index = df.index.map(lambda token: lemmatizer.lemmatize(token))
    return df


if pre_proc == "lemma":
    full_data = lemmatize_df_index(full_data)

# word2idx, idx2word, n_words, words = get_word_to_index_mappings(full_data)

from nltk.corpus import wordnet
import pandas as pd

words = list(wordnet.words())

if regenerate_index:
    word2idx, idx2word, n_words, words = get_word_to_index_mappings(full_data, words)
    tag2idx, idx2tag, n_tags, tags = get_tag_to_index_mappings(full_data)
    char2idx, idx2char, n_chars, chars = get_char_to_index_dict(words)
    [pd.DataFrame(list(index.items())).to_csv(index_name)
     for index, index_name in [(word2idx, 'word2idx'),
                               (char2idx, 'char2idx'),
                               (idx2char, 'idx2char'),
                               (tag2idx, 'tag2idx'),
                               (idx2tag, 'idx2tag')]]
else:
    (word2idx, n_words), (char2idx, n_chars), (idx2tag, n_tags), (tag2idx, _) = \
        [read_index_from_file(index_file) for index_file in ['word2idx', 'char2idx', 'idx2tag', 'tag2idx']]



if pre_proc == "lemma":
    train_data, test_data = lemmatize_df_index(train_data), lemmatize_df_index(test_data)

X_tr = model_instance.process_X(train_data, word2idx, max_sentence_length)
X_te = model_instance.process_X(test_data, word2idx, max_sentence_length)

Y_tr, Y_tr_str = model_instance.process_Y(train_data, tag2idx, max_sentence_length, n_tags)
Y_te, Y_te_str = model_instance.process_Y(test_data, tag2idx, max_sentence_length, n_tags)

max_word_length = np.max([len(word) for word in words])
X_char_tr = get_char_indices(train_data, max_word_length, max_sentence_length, char2idx)
X_char_te = get_char_indices(test_data, max_word_length, max_sentence_length, char2idx)

model = model_instance.get_compiled_model(vectorizer_model_name=vectorizer_model_name,
                                          missing_values_handled=missing_values_handled,
                                          max_sentence_length=max_sentence_length,
                                          max_word_length=max_word_length,
                                          n_words=n_words, n_chars=n_chars, n_tags=n_tags,
                                          word2idx=word2idx)

train_input = [np.array(X_tr), np.array(X_char_tr)] if include_char_embeddings else X_tr
print(train_input)
print(np.array(train_input).shape)
test_input = [X_te, np.array(X_char_te).reshape((len(X_char_te), max_sentence_length, max_word_length))] if include_char_embeddings else np.array(X_te)
history = model.fit(x=X_tr, y=Y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE)

full_model_name = f"{vectorizer_model_name}_{pre_proc}_{missing_values_handled}_e{EPOCHS}_emb{models[vectorizer_model_name]['vector_size']}.h5"

preds = model.predict(test_input)

preds_str, ground_str = get_pred_and_ground_string(Y_test=Y_te, predictions=preds, idx2tag=idx2tag)

assert len(preds_str) == len(ground_str)
report = classification_report(ground_str, preds_str, output_dict=True)

report_file_name = f"{full_model_name}_res.txt"
save_report_to_file(report, vectorizer_model_name=vectorizer_model_name, file_name=report_file_name)

model.save(full_model_name)
model.save_weights(f'{full_model_name}_weights')
