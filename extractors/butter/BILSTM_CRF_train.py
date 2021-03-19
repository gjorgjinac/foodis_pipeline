import os
os.environ["HDF5_DISABLE_VERSION_CHECK"] = "1"
from BILSTM_CRF_model import BILSTMCRFModel

from utils import get_pred_and_ground_string, get_char_indices, \
    get_char_to_index_dict, get_word_to_index_mappings, get_tag_to_index_mappings, \
    read_data_for_task, save_report_to_file
import numpy as np
import tensorflow as tf
import random
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.metrics import classification_report

# --------------SETTINGS-------------

max_sentence_length = 50
EPOCHS = 20
BATCH_SIZE = 256
EMBEDDING = 40
pre_proc = "none"
vectorizer_model_name = 'lexical'
missing_values_handled = True
task_name = "food-classification"

# seeds
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# --------------SETTINGS-------------

full_data, train_data, test_data = read_data_for_task(task_name)

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

model_instance = BILSTMCRFModel()

def lemmatize_df_index(df):
    df.index = df.index.map(lambda token: lemmatizer.lemmatize(token))
    return df


if pre_proc == "lemma":
    full_data = lemmatize_df_index(full_data)


## Global vocabs from full file
word2idx, idx2word, n_words, words = get_word_to_index_mappings(full_data)
tag2idx, idx2tag, n_tags, tags = get_tag_to_index_mappings(full_data)
char2idx, idx2char, n_chars, chars = get_char_to_index_dict(words)

if pre_proc == "lemma":
    train_data, test_data = lemmatize_df_index(train_data), lemmatize_df_index(test_data)


X_tr = model_instance.process_X(train_data, word2idx, max_sentence_length)
X_te = model_instance.process_X(test_data, word2idx, max_sentence_length)

Y_tr, Y_tr_str = model_instance.process_Y(train_data, tag2idx, max_sentence_length, n_tags)
Y_te, Y_te_str = model_instance.process_Y(test_data, tag2idx, max_sentence_length, n_tags)

max_word_length = np.max([len(word) for word in train_data.index])
X_char_tr = get_char_indices(train_data, max_word_length, max_sentence_length, char2idx)
X_char_te = get_char_indices(test_data, max_word_length, max_sentence_length, char2idx)


cbks = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, min_delta = 5e-3)

model = model_instance.get_compiled_model(vectorizer_model_name, missing_values_handled,
                                  max_sentence_length, max_word_length,
                                  n_words,n_chars, n_tags,
                                  word2idx)

history = model.fit(X_tr,
                    Y_tr, epochs=EPOCHS, validation_split=0.1, verbose=1, batch_size=BATCH_SIZE, callbacks=[cbks])

preds = model.predict(X_te)

preds_str, ground_str = get_pred_and_ground_string(Y_test=Y_te, predictions=preds, idx2tag=idx2tag)

assert len(preds_str) == len(ground_str)
report = classification_report(ground_str, preds_str, output_dict=True)
report_file_name = f"{task_name}_{pre_proc}_{missing_values_handled}_e{EPOCHS}_res.txt"
print(report_file_name)
save_report_to_file(report, vectorizer_model_name=vectorizer_model_name, file_name=report_file_name)
