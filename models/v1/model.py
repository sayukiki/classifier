import csv
import os
import json
import math
import numpy as np
import tensorflow as tf
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

        tf.keras.backend.clear_session()

        term_count = len(self.terms.token2id)
        class_count = len(self.classes.token2id)
        depth = 64
        per_sqrt_depth = 1 / 8
        half_depth = 32
        dropout = 0.2
        query_length = self.query_length*2+1

        queries = tf.keras.Input(shape=(query_length), dtype='int32')

        # Token Embedding
        query = tf.keras.layers.Embedding(term_count+1, depth, input_length=query_length, mask_zero=True)(queries)

        # Positional Encoding
        def positional_encoding(x):
            dtype = x.dtype
            batch_size, length, depth = x.shape

            pos = tf.range(0, length)
            pos = tf.expand_dims(pos, 1)
            pos = tf.tile(pos, (1, half_depth))
            pos = tf.cast(pos, dtype)

            pe1 = tf.range(0, depth, delta=2)
            pe1 = tf.expand_dims(pe1, 0)
            pe1 = tf.tile(pe1, (length, 1))
            pe1 = tf.cast(pe1, dtype)

            pe2 = tf.range(1, depth, delta=2)
            pe2 = tf.expand_dims(pe2, 0)
            pe2 = tf.tile(pe2, (length, 1))
            pe2 = tf.cast(pe2, dtype)

            pe1_ = tf.pow(10000.0, (pe1 / depth))
            pe1_ = tf.sin(pos / pe1_)

            pe2_ = tf.pow(10000.0, (pe1 / depth))
            pe2_ = tf.cos(pos / pe2_)

            a = tf.stack([pe1_, pe2_], axis=2)
            a = tf.reshape(a, (length, depth))

            return x + a
        query = tf.keras.layers.Lambda(positional_encoding)(query)

        for _ in range(n_layers):
            # Self-Attention
            q = tf.keras.layers.Dense(depth, use_bias=False)(query) # [batch_size, q_length, depth]
            k = tf.keras.layers.Dense(depth, use_bias=False)(query) # [batch_size, k_length, depth]
            v = tf.keras.layers.Dense(depth, use_bias=False)(query) # [batch_size, v_length, depth]
            logit = tf.keras.layers.Dot(2)([q, k]) # [batch_size, q_length, k_length]
            logit = tf.keras.layers.Lambda(lambda x: x * per_sqrt_depth)(logit)
            attention_weight = tf.keras.layers.Activation('softmax')(logit) # [batch_size, q_length, k_length]
            attention_weight = tf.keras.layers.Dropout(dropout)(attention_weight)
            attention_output = tf.keras.layers.Dot(1)([attention_weight, v]) # [batch_size, q_length, depth]
            attention_output = tf.keras.layers.Dense(depth, use_bias=False)(attention_output) # [batch_size, q_length, depth]
            attention_output = tf.keras.layers.Dropout(dropout)(attention_output)

            # Residual Connection & Layer Normalization
            query = tf.keras.layers.Add()([query, attention_output]) # [batch_size, q_length, depth]
            query = tf.keras.layers.LayerNormalization()(query) # [batch_size, q_length, depth]

            # Position-wise Feed-Forward Networks
            ffn_output = tf.keras.layers.Dense(depth*4, use_bias=True)(query) # [batch_size, q_length, depth*4]
            ffn_output = tf.keras.layers.Activation('relu')(ffn_output)
            ffn_output = tf.keras.layers.Dropout(dropout)(ffn_output)
            ffn_output = tf.keras.layers.Dense(depth, use_bias=True)(ffn_output) # [batch_size, q_length, depth]
            ffn_output = tf.keras.layers.Dropout(dropout)(ffn_output)

            # Residual Connection & Layer Normalization
            query = tf.keras.layers.Add()([query, ffn_output]) # [batch_size, q_length, depth]
            query = tf.keras.layers.LayerNormalization()(query) # [batch_size, q_length, depth]

        # Classifier
        classes = tf.keras.layers.Lambda(lambda x: x[:, :1, :], output_shape=(1, depth))(query) # [batch_size, 1, depth]
        classes = tf.keras.layers.Flatten()(classes) # [batch_size, depth]
        classes = tf.keras.layers.Dense(class_count, use_bias=True)(classes) # [batch_size, classes]
        classes = tf.keras.layers.Activation('softmax')(classes) # [batch_size, classes]

        # TODO Decoder

        model = tf.keras.Model(inputs=[queries], outputs=[classes])
        # print(model.summary())

        return model

    def load(self, name):

        with open(os.path.join('build', '%s-parameters.json'%name), 'r') as f:
            params = json.load(f)

        epochs = params['epochs']
        batch_size = params['batch_size']
        n_layers = params['n_layers']

        self.terms = Dictionary.load(os.path.join('build', '%s-terms.dct'%name))
        self.classes = Dictionary.load(os.path.join('build', '%s-classes.dct'%name))

        self.model = self.get_model(n_layers=n_layers)

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

        # Cross-Validation
        cv_parameters = [
            {'epochs': 100, 'batch_size': 128, 'layers': 1},
            {'epochs': 250, 'batch_size': 128, 'layers': 1},
            {'epochs': 500, 'batch_size': 128, 'layers': 1},
            {'epochs': 1000, 'batch_size': 128, 'layers': 1},
            {'epochs': 2500, 'batch_size': 128, 'layers': 1},
            {'epochs': 100, 'batch_size': 128, 'layers': 2},
            {'epochs': 250, 'batch_size': 128, 'layers': 2},
            {'epochs': 500, 'batch_size': 128, 'layers': 2},
            {'epochs': 1000, 'batch_size': 128, 'layers': 2},
        ]
        n_cv = len(cv_parameters)

        kf = StratifiedKFold(n_splits=5, shuffle=True)
        cv_indexes = []
        for train_indexes, test_indexes in kf.split(data_queries, data_classes):
            cv_indexes.append((train_indexes, test_indexes))

        cv_scores = []
        for i, params in enumerate(cv_parameters):
            epochs = params['epochs']
            batch_size = params['batch_size']
            n_layers = params['n_layers']

            kf_scores = []
            for j, (train_indexes, test_indexes) in enumerate(cv_indexes):
                model = self.get_model(n_layers=n_layers)
                loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
                optimizer = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

                train_queries = data_queries[train_indexes]
                train_classes = data_classes[train_indexes]
                train_batches = math.ceil(len(train_queries)/batch_size)

                def train_step(x, y):
                    with tf.GradientTape() as tape:
                        predictions = model(x)
                        loss = loss_object(y, predictions)
                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                for epoch in range(epochs):
                    for batch in range(train_batches):
                        x = train_queries[batch*batch_size:(batch+1)*batch_size]
                        y = train_classes[batch*batch_size:(batch+1)*batch_size]
                        train_step(x, y)
                        print('\rcv:%d, kf:%d, epoch:%d, batch:%d'%(i+1, j+1, epoch+1, batch+1), end='')

                test_queries = data_queries[test_indexes]
                test_classes = data_classes[test_indexes]
                test_batches = math.ceil(len(test_queries)/batch_size)

                test_loss = tf.keras.metrics.Mean(name='loss')
                test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
                def test_step(x, y):
                    predictions = model(x)
                    test_accuracy(y, predictions)
                for batch in range(test_batches):
                    x = test_queries[batch*batch_size:(batch+1)*batch_size]
                    y = test_classes[batch*batch_size:(batch+1)*batch_size]
                    test_step(x, y)
                accuracy = test_accuracy.result()

                print(', accuracy: %f'%accuracy)
                kf_scores.append(accuracy)

            mean_accuracy = np.mean(kf_scores)
            std_accuracy = np.std(kf_scores)
            print('cv:%d, mean_accuracy: %f, std_accuracy: %f'%(i+1, accuracy, std_accuracy))
            cv_scores.append(accuracy)

        params = cv_parameters[np.argmax(cv_scores)]
        print('choice parameters: %s'%json.dumps(params, ensure_ascii=False))

        # Train
        epochs = params['epochs']
        batch_size = params['batch_size']
        n_layers = params['n_layers']

        self.model = self.get_model(n_layers=n_layers)
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        train_batches = math.ceil(len(data_queries)/batch_size)
        def train_step(x, y):
            with tf.GradientTape() as tape:
                predictions = self.model(x)
                loss = loss_object(y, predictions)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        for epoch in range(epochs):
            for batch in range(train_batches):
                x = data_queries[batch*batch_size:(batch+1)*batch_size]
                y = data_classes[batch*batch_size:(batch+1)*batch_size]
                train_step(x, y)
                print('\rtrain, epoch:%d, batch:%d'%(epoch+1, batch+1), end='')
        print()

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
