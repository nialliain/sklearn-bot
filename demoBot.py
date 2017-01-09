import sys
import re
from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

regexReplacemets = {
    '[TJID]' : '''A[0-9]+''',
}

trainingData = {
    'Hello!': ['hi!', 'hey', 'yo', 'morning!', 'afternoon'],
    'positive': ['yep', 'yes please', 'go ahead', 'please do'],
    'negative': ['no', 'dont', 'do not', 'nope'],
    'No probs!': ['thank you', 'thanks', 'cheers'],
    'blackduck': ['Is blackduck up to date?', 'Can you check Blackduck please?', 'is code centre up to date?'],
    'bd-update': ['update blackduck', 'update code centre'],
    'refdata': ['What is A1', 'look up A1', 'check refdata', 'check reference data'],
}

def blackduck(line):
    return 'Checking, 2 secs..', None

def refdata(line):
    tjids = re.search(regexReplacemets['[TJID]'], line)
    if tjids:
        return 'Checking refdata for ' + tjids.group(0), None
    else:
        return 'Sure - what do you want me to search for?', refdata

responders = {
    'blackduck': blackduck,
    'refdata': refdata,
}

def doRegexReplacement( stringInput ):
    for replacement in regexReplacemets:
        stringInput = re.sub(regexReplacemets[replacement], replacement, stringInput)
    return stringInput

trainingStrings = []
trainingStringCats = []
i=0

for cat in trainingData:
    for trainingString in trainingData[cat]:
        trainingStrings.append(doRegexReplacement(trainingString))
        trainingStringCats.append(i)
    i+=1

chat = Bunch()
chat.data = trainingStrings
chat.target = trainingStringCats
chat.target_names = trainingData.keys()

text_clf = Pipeline([('vect', CountVectorizer(stop_words=['please'])),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, n_iter=5, random_state=42)),
])
_ = text_clf.fit(chat.data, chat.target)
nextFunc = None

while 1:
    try:
        line = sys.stdin.readline()
    except KeyboardInterrupt:
        break
    if not line:
        break
    if nextFunc:
        resp, nextFunc = nextFunc(line)
        print resp
    else:
        categ = chat.target_names[text_clf.predict([re.sub('''A[0-9]*''','[TJID]', line)])[0]]
        if categ in responders:
            resp, nextFunc = responders[categ](line)
            print(resp)
        else:
            print(categ)
