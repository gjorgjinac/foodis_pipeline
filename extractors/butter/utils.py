import os
import re

import pandas as pd
import numpy as np


class SentenceGetter(object):

    def __init__(self, data, label_adapter):
        self.n_sent = 0
        self.data = data
        self.empty = False

        self.grouped = []
        sentence = []
        for key, value in zip(data.iloc[:, 0].keys(), data.iloc[:, 0].values):
            sentence.append((key, label_adapter(value)))
            # print(label_adapter(value))
            if key == '.':
                self.grouped.append(sentence)
                sentence = []
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.sentences[self.n_sent]
            self.n_sent += 1
            return s
        except:
            return None


def save_report_to_file(report, vectorizer_model_name, file_name, nn_model_name='BILSTM_CharEmb'):
    base = os.path.join("results", nn_model_name, vectorizer_model_name)
    os.makedirs(base, exist_ok=True)

    ret = pd.DataFrame.from_dict(report)
    ltx = ret.to_latex(label="tab:results", caption="Results")
    print(ret)
    with open(os.path.join(base, file_name), "w") as f_out:
        f_out.write(f"{ltx}\n")


def get_pred_and_ground_string(Y_test, predictions, idx2tag):
    preds_str = []
    ground_str = []

    len_dict = dict()
    for idx1, el in enumerate(Y_test):
        for idx2, el2 in enumerate(el):
            j = idx2tag[np.argmax(el2) if len(Y_test.shape) >= 3 else el2]
            if j == 'PAD':
                len_dict[idx1] = idx2
                break
            ground_str.append(j)

    for idx1, el in enumerate(predictions):
        for idx2, el2 in enumerate(el):
            j = idx2tag[np.argmax(el2) if len(predictions.shape) >= 3 else el2]
            if idx2 >= len_dict[idx1]:
                break
            preds_str.append(j)

    return preds_str, ground_str


def transform_label(l):
    # print(re.search(r'^([BI]).*', l))
    return re.sub(r'^([BI]).*', r'\1-FOOD', l)


def get_label(l):
    return l


def get_char_indices(data, max_len_char, max_sentence_length, char2idx, data_as_sentences = False):
    sentences = data if data_as_sentences else \
        [[word_label[0] for word_label in sentence] for sentence in SentenceGetter(data, label_adapter=get_label).sentences]
    X_char = []
    for sentence in sentences:
        sent_seq = []
        for i in range(max_sentence_length):
            word_seq = []
            for j in range(max_len_char):
                try:
                    word_seq.append(int(char2idx.get(sentence[i][j])))
                except:
                    word_seq.append(int(char2idx.get("PAD")))

            sent_seq.append(word_seq)
        X_char.append(np.array(sent_seq))
    return np.array(X_char)


def get_embedding_weights(vectorizer_model_name, vectorizer_model_size, missing_values_handled, word2idx):
    bpath = "./vectors"
    vectors_path = os.path.join(bpath, f'missing_values_handled_{missing_values_handled}/{vectorizer_model_name}')
    vectors = pd.read_csv(vectors_path, index_col=[0])
    embedding_weights = [vectors.loc[word.lower(), :] if word in vectors.index else np.zeros(vectorizer_model_size) for
                         word, index in word2idx.items()]
    embedding_weights = np.array(embedding_weights)
    print(embedding_weights.shape)
    return embedding_weights


def get_char_to_index_dict(words):
    chars = set([w_i for w in words for w_i in w])
    char2idx = {c: i + 2 for i, c in enumerate(chars)}
    char2idx["UNK"] = 1
    char2idx["PAD"] = 0
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char, len(char2idx.keys()), chars


def get_word_to_index_mappings(full_data, words = None):
    if words is None:
        words = set()
    words = list(set(words).union(set(full_data.iloc[:, 0].index)))
    word2idx = {w: i + 2 for i, w in enumerate(words)}
    word2idx["UNK"] = 1  # Unknown words
    word2idx["PAD"] = 0  # Padding
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word, len(word2idx.keys()), words


def get_tag_to_index_mappings(full_data):
    tags = list(set(full_data.iloc[:, 0].values))
    tag2idx = {t: i + 1 for i, t in enumerate(tags)}
    tag2idx["PAD"] = 0
    idx2tag = {i: w for w, i in tag2idx.items()}
    return tag2idx, idx2tag, len(tag2idx.keys()), tags


def read_df_from_tsv_file(file_path):
    return pd.read_csv(file_path, encoding="latin1", delimiter='\t').fillna(method="ffill")


def read_data_for_task(task_name, bpath="data"):
    full_path = os.path.join(bpath, f"full-{task_name}.txt")
    train_path = os.path.join(bpath, f"train-{task_name}.txt")
    test_path = os.path.join(bpath, f"test-{task_name}.txt")
    return read_df_from_tsv_file(full_path), \
           read_df_from_tsv_file(train_path), \
           read_df_from_tsv_file(test_path)


def pad_string_matrix(string_matrix, max_len, pad_value='__PAD__'):
    padded_matrix = []
    for seq in string_matrix:
        new_seq = []
        for i in range(max_len):
            try:
                new_seq.append(seq[i])
            except:
                new_seq.append(pad_value)
        padded_matrix.append(new_seq)
    return padded_matrix

def read_index_from_file(file_name):
    index = pd.read_csv(file_name, index_col=[0])
    index = {t[0]: t[1] for t in list(index.values)}
    return index, len(index.keys())