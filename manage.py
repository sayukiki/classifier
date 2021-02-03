from models.v1 import Classifier as ClassifierV1

def build_classifier(name, data):

    return ClassifierV1().build(name, data)

def load_classifier(name):

    return ClassifierV1().load(name)

if __name__ == '__main__':

    classifier = build_classifier('it-helpdesk-20210109', 'dataset/it-helpdesk-20210109.csv')

    classifier = load_classifier('it-helpdesk-20210109')

    print(classifier.predict('パソコンが壊れた'))
    print(classifier.predict('携帯電話がつながらない'))
    print(classifier.predict('音が出なくなった'))
