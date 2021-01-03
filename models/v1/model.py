import csv
import os
import json
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Embedding, Activation, Add, Dot, Lambda, Flatten, Dropout, LayerNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from gensim.corpora import Dictionary
from sklearn.model_selection import StratifiedKFold 

class Classifier():

    def __init__(self):

        self.weights = None
        self.terms = None
        self.classes = None
        self.query_length = 140

    def get_terms(self, text, padding=None):

        texts = text.split(' ')

        terms = []

        bi_gram = [text[i : i+2] for text in texts for i in range(len(text)-1)]
        if padding:
            bi_gram.extend([padding]*self.query_length)
        terms.extend(bi_gram[:self.query_length])

        tri_gram = [text[i : i+3] for text in texts for i in range(len(text)-2)]
        if padding:
            tri_gram.extend([padding]*self.query_length)
        terms.extend(tri_gram[:self.query_length])

        return terms

    def get_model(self, n_layers=1):

        K.clear_session()

        term_count = len(self.terms.token2id)
        class_count = len(self.classes.token2id)
        depth = 64
        per_sqrt_depth = 1 / 8
        half_depth = 32
        dropout = 0.2
        query_length = self.query_length*2+1

        queries = Input(shape=(query_length), dtype='int32')

        # Token Embedding
        query = Embedding(term_count+1, depth, input_length=query_length, mask_zero=True)(queries)

        # Positional Encoding
        def positional_encoding(x):
            dtype = x.dtype
            batch_size, length, depth = x.shape

            pos = K.arange(0, stop=length)
            pos = K.expand_dims(pos, 1)
            pos = K.tile(pos, (1, half_depth))
            pos = K.cast(pos, dtype)

            pe1 = K.arange(0, stop=depth, step=2)
            pe1 = K.expand_dims(pe1, 0)
            pe1 = K.tile(pe1, (length, 1))
            pe1 = K.cast(pe1, dtype)

            pe2 = K.arange(1, stop=depth, step=2)
            pe2 = K.expand_dims(pe2, 0)
            pe2 = K.tile(pe2, (length, 1))
            pe2 = K.cast(pe2, dtype)

            pe1_ = K.pow(10000.0, (pe1 / depth))
            pe1_ = K.sin(pos / pe1_)

            pe2_ = K.pow(10000.0, (pe1 / depth))
            pe2_ = K.cos(pos / pe2_)

            a = K.stack([pe1_, pe2_], axis=2)
            a = K.reshape(a, (length, depth))

            return x + a
        query = Lambda(positional_encoding)(query)

        for _ in range(n_layers):
            # Self-Attention
            q = Dense(depth, use_bias=False)(query) # [batch_size, q_length, depth]
            k = Dense(depth, use_bias=False)(query) # [batch_size, k_length, depth]
            v = Dense(depth, use_bias=False)(query) # [batch_size, v_length, depth]
            logit = Dot(2)([q, k]) # [batch_size, q_length, k_length]
            logit = Lambda(lambda x: x * per_sqrt_depth)(logit)
            attention_weight = Activation('softmax')(logit) # [batch_size, q_length, k_length]
            attention_weight = Dropout(dropout)(attention_weight)
            attention_output = Dot(1)([attention_weight, v]) # [batch_size, q_length, depth]
            attention_output = Dense(depth, use_bias=False)(attention_output) # [batch_size, q_length, depth]
            attention_output = Dropout(dropout)(attention_output)

            # Residual Connection & Layer Normalization
            query = Add()([query, attention_output]) # [batch_size, q_length, depth]
            query = LayerNormalization()(query) # [batch_size, q_length, depth]

            # Position-wise Feed-Forward Networks
            ffn_output = Dense(depth*4, use_bias=True)(query) # [batch_size, q_length, depth*4]
            ffn_output = Activation('relu')(ffn_output)
            ffn_output = Dropout(dropout)(ffn_output)
            ffn_output = Dense(depth, use_bias=True)(ffn_output) # [batch_size, q_length, depth]
            ffn_output = Dropout(dropout)(ffn_output)

            # Residual Connection & Layer Normalization
            query = Add()([query, ffn_output]) # [batch_size, q_length, depth]
            query = LayerNormalization()(query) # [batch_size, q_length, depth]

        # Classifier
        classes = Lambda(lambda x: x[:, :1, :], output_shape=(1, depth))(query) # [batch_size, 1, depth]
        classes = Flatten()(classes) # [batch_size, depth]
        classes = Dense(class_count, use_bias=True)(classes) # [batch_size, classes]
        classes = Activation('softmax')(classes) # [batch_size, classes]

        # TODO Decoder

        model = Model(inputs=[queries], outputs=[classes])
        # print(model.summary())

        return model

    def load(self, name):

        with open(os.path.join('build', '%s-parameters.json'%name), 'r') as f:
            parameters = json.load(f)

        self.terms = Dictionary.load(os.path.join('build', '%s-terms.dct'%name))
        self.classes = Dictionary.load(os.path.join('build', '%s-classes.dct'%name))

        self.model = self.get_model(n_layers=parameters['layers'])

        self.model.load_weights(os.path.join('build', '%s-weights.h5'%name))

        return self

    def build(self, name, data):

        self.terms = Dictionary([['<CLS>']])
        self.classes = Dictionary()

        with open(data) as f:
            reader = csv.reader(f)
            for row in reader:
                text = row[0]
                clas = row[1]

                terms = self.get_terms(text)

                self.terms.add_documents([terms])
                self.classes.add_documents([[clas]])

        data_queries = []
        data_classes = []

        with open(data) as f:
            reader = csv.reader(f)
            for row in reader:
                text = row[0]
                clas = row[1]

                terms = ['<CLS>']
                terms.extend(self.get_terms(text, padding='<PAD>'))
                terms = self.terms.doc2idx(terms, unknown_word_index=-1)
                terms = list(map(lambda x: x+1, terms))

                clas = self.classes.token2id[clas]

                data_queries.append(terms)
                data_classes.append(clas)

        data_queries = np.array(data_queries)
        data_classes = np.array(data_classes)

        cv_parameters = [
            {'epochs': 100, 'layers': 1}, # accuracy: 0.143201 (std 0.004911)
            {'epochs': 250, 'layers': 1}, # accuracy: 0.482982 (std 0.275570)
            {'epochs': 500, 'layers': 1}, # accuracy: 0.812202 (std 0.044028)
            {'epochs': 1000, 'layers': 1}, # accuracy: 0.791053 (std 0.053435)
            {'epochs': 2500, 'layers': 1}, # accuracy: 0.786430 (std 0.043388)
            {'epochs': 100, 'layers': 2},
            {'epochs': 250, 'layers': 2},
            {'epochs': 500, 'layers': 2}, # accuracy: 0.805144 (std 0.009774)
            {'epochs': 1000, 'layers': 2},
        ]
        n_cv = len(cv_parameters)

        n_kf = 5
        cv_indexes = []
        for train_indexes, test_indexes in StratifiedKFold(n_splits=n_kf, shuffle=True).split(data_queries, data_classes):
            cv_indexes.append((train_indexes, test_indexes))

        cv_scores = []
        for i, params in enumerate(cv_parameters, 1):
            kf_scores = []
            for j, (train_indexes, test_indexes) in enumerate(cv_indexes, 1):
                self.model = self.get_model(n_layers=params['layers'])
                self.model.compile(optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.98, epsilon=1e-9), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                self.model.fit(data_queries[train_indexes], data_classes[train_indexes], epochs=params['epochs'], batch_size=128, verbose=0)
                scores = self.model.evaluate(data_queries[test_indexes], data_classes[test_indexes], verbose=0)
                accuracy = scores[1]
                kf_scores.append(accuracy)
                print('kf progress: %d/%d %d/%d accuracy: %f'%(i, n_cv, j, n_kf, accuracy))
            print('cv progress: %d/%d accuracy: %f (std %f)'%(i, n_cv, np.mean(kf_scores), np.std(kf_scores)))
            cv_scores.append(np.mean(kf_scores))

        params = cv_parameters[np.argmax(cv_scores)]

        self.model = self.get_model(n_layers=params['layers'])
        self.model.compile(optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.98, epsilon=1e-9), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(data_queries, data_classes, epochs=params['epochs'], batch_size=128, verbose=0)
        self.model.save_weights(os.path.join('build', '%s-weights.h5'%name))
        self.terms.save(os.path.join('build', '%s-terms.dct'%name))
        self.classes.save(os.path.join('build', '%s-classes.dct'%name))
        with open(os.path.join('build', '%s-parameters.json'%name), 'w') as f:
            json.dump(params, f)

        return self

    def predict(self, text):

        terms = ['<CLS>']
        terms.extend(self.get_terms(text, padding='<PAD>'))
        terms = self.terms.doc2idx(terms, unknown_word_index=-1)
        terms = list(map(lambda x: x+1, terms))

        result = self.model.predict([terms])[0]
        result = result.astype(float)
        scores = np.sort(result)[::-1]
        preds = np.argsort(result)[::-1]

        return [(self.classes[pred], score) for pred, score in list(zip(preds, scores))[:5]]
