import copy

import numpy as np
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Input, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tf2crf import CRF

from .model_definition import models
from .utils import get_embedding_weights, SentenceGetter, get_label


class BILSTMCRFModel:

    def get_compiled_model(self, vectorizer_model_name, missing_values_handled,
                           max_sentence_length, max_word_length,
                           n_words, n_chars, n_tags,
                           word2idx):

        vectorizer_model_settings = models[vectorizer_model_name]
        vectorizer_model_size = vectorizer_model_settings['vector_size']

        word_in =Input(shape=(None,), dtype='int32')
        if not vectorizer_model_settings['precomputed_vectors']:
            emb_word = Embedding(input_dim=n_words, output_dim=vectorizer_model_size,
                                 input_length=max_sentence_length,
                                 mask_zero=True)(word_in)
        else:
            embedding_weights = get_embedding_weights(vectorizer_model_name, vectorizer_model_size,
                                                      missing_values_handled,
                                                      word2idx)
            emb_word = Embedding(input_dim=n_words, output_dim=vectorizer_model_size,
                                 input_length=max_sentence_length,
                                 mask_zero=True,
                                 weights=[embedding_weights], trainable=False)(word_in)

        model = Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.1))(emb_word)
        model = TimeDistributed(Dense(50, activation='relu'))(model)
        crf = CRF(n_tags)
        out = crf(model)

        model = Model(word_in, out)
        model.summary()
        model.compile(optimizer="rmsprop", loss=crf.loss, metrics=[crf.accuracy])
        return model


    def process_X(self, data, word2idx, max_sentence_length, data_as_sentences = False):
        if data_as_sentences:
            X = [[word2idx[w] if w in word2idx.keys() else word2idx['UNK'] for w in s] for s in data]
        else:
            sentences = SentenceGetter(data, label_adapter=get_label).sentences
            X = [[word2idx[w[0]] if w[0] in word2idx.keys() else word2idx['UNK'] for w in s] for s in sentences]

        X = pad_sequences(maxlen=max_sentence_length, sequences=X, padding="post", value=word2idx["PAD"])
        return X

    def process_Y(self, data, tag2idx, max_sentence_length, n_tags):
        sentence_getter = SentenceGetter(data, label_adapter=get_label)
        Y = [[tag2idx[w[1]] for w in s] for s in sentence_getter.sentences]
        Y_str = copy.deepcopy(Y)
        Y = pad_sequences(maxlen=max_sentence_length, sequences=Y, padding="post", value=tag2idx["PAD"])
        Y = np.array([to_categorical(i, num_classes=n_tags + 1) for i in Y])  # n_tags+1(PAD)
        return Y, Y_str
