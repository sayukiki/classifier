from .v1 import Classifier as ClassifierV1

classifiers = {
    'it-helpdesk-20210109': ClassifierV1().load('it-helpdesk-20210109'),
}
