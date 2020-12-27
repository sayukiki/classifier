import os
import pickle
from keras.models import load_model
from gensim.corpora import Dictionary
from v1 import Classifier as ClassifierV1

def build_classifier(name, data):

    return ClassifierV1().build(name, data)

def load_classifier(name):

    return ClassifierV1().load(name)

if __name__ == '__main__':

    # classifier = build_classifier('it-helpdesk-20201227', 'dataset/it-helpdesk.csv')

    classifier = load_classifier('it-helpdesk-20201227')

    print(classifier.predict('パソコンが壊れた'))
