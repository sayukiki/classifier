import csv
import os
import numpy as np
from keras.layers import Input, Dense, Embedding, Activation, Add, Dot, Lambda, Flatten, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.backend import shape, arange, sin, cos, cast, pow, stack, flatten, tile, transpose, reshape, expand_dims
from gensim.corpora import Dictionary

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

    def get_model(self):

        term_count = len(self.terms.token2id)
        class_count = len(self.classes.token2id)
        depth = 64
        half_depth = 32

        queries = Input(shape=(self.query_length*2+1,), dtype='int32')

        # Token Embedding
        query = Embedding(term_count+1, depth, input_length=self.query_length*2+1, mask_zero=True)(queries)

        # Positional Encoding
        def positional_encoding(x):
            dtype = x.dtype
            batch_size, length, depth = x.shape

            pos = arange(0, stop=length)
            pos = expand_dims(pos, 1)
            pos = tile(pos, (1, half_depth))
            pos = cast(pos, dtype)

            pe1 = arange(0, stop=depth, step=2)
            pe1 = expand_dims(pe1, 0)
            pe1 = tile(pe1, (length, 1))
            pe1 = cast(pe1, dtype)

            pe2 = arange(1, stop=depth, step=2)
            pe2 = expand_dims(pe2, 0)
            pe2 = tile(pe2, (length, 1))
            pe2 = cast(pe2, dtype)

            pe1_ = pow(10000.0, (pe1 / depth))
            pe1_ = sin(pos / pe1_)

            pe2_ = pow(10000.0, (pe1 / depth))
            pe2_ = cos(pos / pe2_)

            a = stack([pe1_, pe2_], axis=2)
            a = reshape(a, (length, depth))

            return x + a
        query = Lambda(positional_encoding)(query)

        # Self-Attention
        q = Dense(depth, use_bias=False)(query) # [batch_size, q_length, depth]
        k = Dense(depth, use_bias=False)(query) # [batch_size, k_length, depth]
        v = Dense(depth, use_bias=False)(query) # [batch_size, v_length, depth] k_length=v_length
        logit = Dot(1)([q, k]) # [batch_size, q_length, k_length]
        attention_weight = Activation('softmax')(logit) # [batch_size, q_length, k_length]
        attention_weight = Dropout(0.2)(attention_weight)
        attention_output = Dot((2, 1))([k, attention_weight]) # [batch_size, q_length, depth]
        attention_output = Dense(depth, use_bias=False)(attention_output) # [batch_size, q_length, depth]
        attention_output = Dropout(0.2)(attention_output)
        attention_output = Add()([query, attention_output]) # [batch_size, q_length, depth]
        query = attention_output # [batch_size, q_length, depth]

        # Feedforward Network
        ffn = Dense(depth*4, use_bias=True, activation='relu')(query) # [batch_size, q_length, depth*4]
        ffn = Dropout(0.2)(ffn)
        ffn = Dense(depth, use_bias=True)(ffn) # [batch_size, q_length, depth]
        ffn = Dropout(0.2)(ffn)
        ffn = Add()([query, ffn]) # [batch_size, q_length, depth]
        query = ffn # [batch_size, q_length, depth]

        # Classifier
        classes = Lambda(lambda x: x[:, :1, :], output_shape=(1,))(query) # [batch_size, 1, depth]
        classes = Flatten()(classes) # [batch_size, depth]
        classes = Dense(class_count, use_bias=True)(classes) # [batch_size, classes]
        classes = Activation('softmax')(classes) # [batch_size, classes]

        # TODO Decoder

        model = Model(inputs=[queries], outputs=[classes])
        # print(model.summary())

        return model

    def load(self, name):

        self.terms = Dictionary.load(os.path.join('build', '%s-terms.dct'%name))
        self.classes = Dictionary.load(os.path.join('build', '%s-classes.dct'%name))

        self.model = self.get_model()

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

        self.model = self.get_model()

        self.model.compile(optimizer=Adam(lr=0.0001, beta_2=0.98), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(data_queries, data_classes, epochs=5000, batch_size=128)

        self.model.save_weights(os.path.join('build', '%s-weights.h5'%name))
        self.terms.save(os.path.join('build', '%s-terms.dct'%name))
        self.classes.save(os.path.join('build', '%s-classes.dct'%name))

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
