import os

from config import global_output_directory_name

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import wordnet
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow import keras


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


def read_index_from_file(file_name, file_dir=''):
    index = pd.read_csv(f'{file_dir}/{file_name}', index_col=[0])
    index = {t[0]: t[1] for t in list(index.values)}
    return index, len(index.keys())


"""### Utility Functions"""

import re


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
            if key is '.':
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
            j = idx2tag[np.argmax(el2)]
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


def get_char_indices(data, max_len_char, max_sentence_length, char2idx):
    sentence_getter = SentenceGetter(data, label_adapter=get_label)
    X_char = []
    for sentence in sentence_getter.sentences:
        sent_seq = []
        for i in range(max_sentence_length):
            word_seq = []
            for j in range(max_len_char):
                try:
                    word_seq.append(char2idx.get(sentence[i][0][j]))
                except:
                    word_seq.append(char2idx.get("PAD"))
            sent_seq.append(word_seq)
        X_char.append(np.array(sent_seq))
    return X_char


def get_embedding_weights(vectorizer_model_name, vectorizer_model_size, missing_values_handled, word2idx):
    bpath = "vectors"
    vectors_path = os.path.join(bpath, f'missing_values_handled_{missing_values_handled}/{vectorizer_model_name}')
    vectors = pd.read_csv(vectors_path, index_col=[0])
    embedding_weights = [vectors.loc[word.lower(), :] if word in vectors.index else np.zeros(vectorizer_model_size) for
                         word, index in word2idx.items()]
    embedding_weights = np.array(embedding_weights)
    print(embedding_weights.shape)
    return embedding_weights


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


"""### Model definition"""

import numpy as np
import copy
import os

from tf2crf import CRF
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Bidirectional, Dropout
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical

from transformers import TFBertPreTrainedModel, TFBertMainLayer


class FoodNERBertForTokenClassification(TFBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels
        self.config = config
        self.bert = TFBertMainLayer(self.config, name="bert")
        self.bilstm = Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.1))
        self.dropout = Dropout(0.2)
        self.time_distributed = TimeDistributed(Dense(self.num_labels, activation="relu"))
        self.crf = CRF(self.num_labels + 1)

    def call(self, inputs, **kwargs):
        outputs = self.bert(inputs, **kwargs)
        sequence_output = outputs[0]
        bilstm = self.bilstm(sequence_output)
        dropout = self.dropout(bilstm)
        td = self.time_distributed(dropout)
        return self.crf(td)


class BERTCRFModel():
    def get_compiled_model(self, model_to_load, n_tags, full_finetuning=False):
        model = FoodNERBertForTokenClassification.from_pretrained(
            model_to_load,
            num_labels=n_tags + 1,
            output_attentions=False,
            output_hidden_states=False)

        model.summary()
        # optimizer = Adam(learning_rate=3e-5, epsilon=1e-8)
        optimizer = RMSprop(learning_rate=3e-5, epsilon=1e-8)

        model.compile(optimizer=optimizer, loss=model.crf.loss, metrics=[model.crf.accuracy])
        # model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["acc"])
        return model

    def process_X(self, data, word2idx, max_sentence_length, tag2idx, sentences=None):
        sentence_getter = SentenceGetter(data, label_adapter=get_label)
        if sentences == None:
            sentences = sentence_getter.sentences
        indices = [[word2idx[w[0]] if w[0] in word2idx.keys() else word2idx['UNK'] for w in s] for s in sentences]
        indices = pad_sequences(maxlen=max_sentence_length, sequences=indices, padding="post", value=word2idx["PAD"])
        attention_masks = [[float(i != word2idx["PAD"]) for i in ii] for ii in indices]
        return indices, attention_masks, sentences

    def process_Y(self, data, tag2idx, max_sentence_length, n_tags):
        sentence_getter = SentenceGetter(data, label_adapter=get_label)
        Y = [[tag2idx[w[1]] for w in s] for s in sentence_getter.sentences]
        Y_str = copy.deepcopy(Y)
        Y = pad_sequences(maxlen=max_sentence_length, sequences=Y, padding="post", value=tag2idx["PAD"])
        Y = np.array([to_categorical(i, num_classes=n_tags + 1) for i in Y])
        return Y, Y_str

def example_to_features(input_ids, attention_masks, label_ids):
    return {"input_ids": input_ids, "attention_mask": attention_masks}, label_ids
"""### Model training"""

def train():
    batch_size = 32
    epochs = 1000
    max_sentence_length = 50
    regenerate_index=False
    words = list(wordnet.words())
    tasks = ['food-classification']
    print(len(words))
    stored_models = {
        'bert': 'bert-base-cased',
        # 'bioBert-standard': '/content/drive/My Drive/Colab Notebooks/data/biobert',
        # 'bioBert-large': '/content/drive/My Drive/Colab Notebooks/data/biobert_large'
    }
    for task in tasks:

        model_instance = BERTCRFModel()

        full_data, train_data, test_data = read_data_for_task(task,'data')

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

        tr_tags, _ = model_instance.process_Y(train_data, tag2idx, max_sentence_length, n_tags)
        te_tags, _ = model_instance.process_Y(test_data, tag2idx, max_sentence_length, n_tags)

        tr_inputs, tr_masks, _ = model_instance.process_X(train_data, word2idx, max_sentence_length, tag2idx)
        te_inputs, te_masks, _ = model_instance.process_X(test_data, word2idx, max_sentence_length, tag2idx)

        tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(tr_inputs, tr_tags, random_state=2018,
                                                                    test_size=0.1)
        tr_masks, val_masks, _, _ = train_test_split(tr_masks, tr_masks, random_state=2018, test_size=0.1)

        train_ds = tf.data.Dataset.from_tensor_slices((tr_inputs, tr_masks, tr_tags)).map(example_to_features).shuffle(
            buffer_size=1000).batch(batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((te_inputs, te_masks, te_tags)).map(
            example_to_features)  # .shuffle(buffer_size=1000).batch(batch_size)
        val_ds = tf.data.Dataset.from_tensor_slices((val_inputs, val_masks, val_tags)).map(example_to_features).shuffle(
            buffer_size=1000).batch(batch_size)

        for model_prefix, load_model in stored_models.items():
            model_name = f'bert_crf_e{epochs}'
            cbks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, min_delta=5e-3)
            model = model_instance.get_compiled_model(load_model, n_tags)
            history = model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[cbks])

            model.save(model_name)
            preds = model.predict(te_inputs)
            preds_str, ground_str = get_pred_and_ground_string(Y_test=te_tags, predictions=preds, idx2tag=idx2tag)
            print(preds_str)
            assert len(preds_str) == len(ground_str)

            report = classification_report(ground_str, preds_str, output_dict=True)
            print(report)
            report_file_name = f"{task}_e{epochs}_res.txt"
            save_report_to_file(report, vectorizer_model_name=model_name, file_name=report_file_name)

    print(preds_str)


from nltk import word_tokenize
import nltk
import spacy
import pandas as pd


def extract_from_dataset(dataset, epochs, path_to_module):
    model_name = f'bert_crf_e{epochs}'
    model_name = os.path.join(path_to_module, model_name)
    nltk.download('punkt')
    abstract_df = pd.read_csv(os.path.join('inputs',dataset, 'abstracts.csv'), index_col=[0]).fillna('')
    print(abstract_df.columns)
    (word2idx, n_words), (char2idx, n_chars), (idx2tag, n_tags), (tag2idx, _) = \
        [read_index_from_file(index_file, path_to_module) for index_file in ['word2idx', 'char2idx', 'idx2tag', 'tag2idx']]
    max_sentence_length=50
    spacy_model = spacy.load('en_core_web_sm')
    entities_in_all_abstracts = []
    model_instance = BERTCRFModel()
    i = 0
    iob_tags = []
    iob_words = []
    iob_file_names = []
    iob_sentence_ids = []
    model = model_instance.get_compiled_model('bert-base-cased',n_tags)
    model.load_weights(os.path.join(model_name, 'variables','variables'))
    for index, row in abstract_df.iterrows():
        abstract = row['abstract']
        abstract_id = row['PMID']
        # print(f'{i}/{abstract_df.shape[0]}')
        i += 1
        doc = spacy_model(abstract)
        abstract_words = word_tokenize(abstract)
        if len(abstract_words) == 0:
            continue
        apply_df = pd.DataFrame(abstract_words, index=abstract_words)
        sentences_as_list = [[(t.text, t.text) for t in s] for s in doc.sents]
        sentences_as_texts = [s.text for s in doc.sents]
        assert len(sentences_as_texts) == len(sentences_as_list)
        apply_df, _, apply_sentences = model_instance.process_X(apply_df, word2idx, max_sentence_length, tag2idx)
        print(apply_df.min())
        print(apply_df.max())
        predictions = model.predict(apply_df)

        food_pieces = []
        apply_sentences_as_words = [[word_label_tuple[0] for word_label_tuple in sentence] for sentence in
                                    apply_sentences]

        predictions = [sentence_predictions[0:len(apply_sentences_as_words[sentence_index])] for
                       sentence_index, sentence_predictions in enumerate(predictions)]

        print(len(iob_tags))
        print(len(iob_words))

        for sentence_index, sentence_predictions in enumerate(predictions):
            food_entities = []
            for word_index, word_prediction in enumerate(sentence_predictions):
                word = apply_sentences_as_words[sentence_index][word_index]
                label = idx2tag[word_prediction]
                iob_tags.append(label)
                iob_words.append(word)
                iob_sentence_ids.append(sentence_index)
                iob_file_names.append(abstract_id)
                if label == 'B-FOOD':
                    food_pieces.append(word)
                elif label == 'I-FOOD':  # and len(food_pieces) > 0:
                    food_pieces.append(word)
                else:
                    if len(food_pieces) > 0:
                        food_entities.append(' '.join(food_pieces))
                        food_pieces = []
            if len(food_entities) > 0:
                sentence_text = ' '.join('')
                entities = [
                    (-1, -1, 'food', '', entity, sentences_as_texts[sentence_index], sentence_index, abstract_id) for
                    entity in food_entities]
                # entities = [(span.start_char, span.end_char, 'food','', span.text, span.sent.text.strip(),
                #                 list(doc.sents).index(span.sent)) if span.sent in doc.sents else None for span in extracted_entities]
                entities_in_all_abstracts += entities
                print(entities)

    entities_columns = ['start_char', 'end_char', 'entity_type', 'entity_id', 'text', 'sentence', 'sentence_index',
                        'file_name']
    entities_df = pd.DataFrame(entities_in_all_abstracts, columns=entities_columns)
    entities_df['extractor'] = 'foodner'
    entities_df.to_csv( f'{global_output_directory_name}/{dataset}/bert_crf_foods.csv')

    iob_df = pd.DataFrame(
        {'word': iob_words, 'tag': iob_tags, 'file_name': iob_file_names, 'sentence_index': iob_sentence_ids})
    print(iob_df)
    iob_df.to_csv(os.path.join('results', f'bert_{dataset}_iob.csv'))


if __name__ == '__main__':
    extract_from_dataset('stratified_1000', 1000, '')
