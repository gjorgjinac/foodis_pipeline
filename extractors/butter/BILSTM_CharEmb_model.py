import os
from keras_preprocessing.sequence import pad_sequences
import numpy as np
from .model_definition import models
import copy
from .utils import get_embedding_weights, SentenceGetter, get_label
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Input, Bidirectional, concatenate, SpatialDropout1D
from tensorflow.keras.utils import to_categorical

class BILSTMDoubleInputModel:

    def get_compiled_model(self, vectorizer_model_name, missing_values_handled,
                           max_sentence_length, max_word_length,
                           n_words, n_chars, n_tags,
                           word2idx):

        vectorizer_model_settings = models[vectorizer_model_name]
        vectorizer_model_size = vectorizer_model_settings['vector_size']
        print(vectorizer_model_size)
        word_in = Input(shape=(max_sentence_length,))
        if not vectorizer_model_settings['precomputed_vectors']:
            emb_word = Embedding(input_dim=n_words , output_dim=vectorizer_model_size,
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

        # input and embeddings for characters
        char_in = Input(shape=(max_sentence_length, max_word_length,))
        emb_char = TimeDistributed(Embedding(input_dim=n_chars, output_dim=10, mask_zero=True))(char_in)
        char_enc = TimeDistributed(LSTM(units=20, return_sequences=False,
                                        recurrent_dropout=0.5))(emb_char)

        model = concatenate([emb_word, char_enc])
        model = Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.1))(model)
        model = TimeDistributed(Dense(50, activation='relu'))(model)
        model = Dense(n_tags, activation=None)(model)
        crf = CRF(dtype='float32', name='crf')
        output = crf(model)
        base_model = Model(word_in, output)
        base_model.compile(optimizer='adam')
        model = ModelWithCRFLoss(base_model)
        model.compile(optimizer='adam')
        model.summary()
        return model


    def process_X(self, data, word2idx, max_sentence_length, data_as_sentences = False):
        if data_as_sentences:
            X = [[word2idx[w] if w in word2idx.keys() else word2idx['UNK'] for w in s] for s in data]
        else:
            sentences = SentenceGetter(data, label_adapter=get_label).sentences
            X = [[word2idx[w[0]] if w[0] in word2idx.keys() else word2idx['UNK'] for w in s] for s in sentences]
        X = pad_sequences(maxlen=max_sentence_length, sequences=X, padding="post", value=word2idx["PAD"])
        return np.array(X)

    def process_Y(self, data, tag2idx, max_sentence_length, n_tags):
        sentence_getter = SentenceGetter(data, label_adapter=get_label)
        Y = [[tag2idx[w[1]] for w in s] for s in sentence_getter.sentences]
        Y_str = copy.deepcopy(Y)
        Y = pad_sequences(maxlen=max_sentence_length, sequences=Y, padding="post", value=tag2idx["PAD"])
        Y = np.array([to_categorical(i, num_classes=n_tags + 1) for i in Y])  # n_tags+1(PAD)
        return Y, Y_str
