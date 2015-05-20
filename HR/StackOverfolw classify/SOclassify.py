import json
import re

splitter = re.compile('[^a-zA-Z0-9]')
data = []
skip = True
nobj = 0
with open('training.json') as f:
    for line in f:
        if skip == False:
            data.append(json.loads(line))
        else:
            nobj = int(line.strip())
            skip = False

"""
print data[1]['topic']
print data[1]['question']
print data[1]['excerpt']
print nobj
"""

traindata = []
trainlabel = []

for i in range(0, nobj):
    d = data[i]['question'].rstrip('\n') +' '+data[i]['excerpt'].rstrip('\n')
    #line = splitter.split(d)
    #line = ' '.join(line)
    d= splitter.sub(' ',d)
    traindata.append(d)
    trainlabel.append(data[i]['topic'])

#print traindata[0]
#print trainlabel[0]


# create count word vector
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(traindata)
#print X_train_counts.shape

# Create tf-idf
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#print X_train_tfidf.shape


#read test data here
testdata =[]
ntest = input()
for t in range(0, ntest):
    s = raw_input()
    obj = json.loads(s)
    d = obj['question'] + ' ' + obj['excerpt']
    d = splitter.sub(' ',d)
    testdata.append(d)

X_new_counts = count_vect.transform(testdata)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

from sklearn.linear_model import PassiveAggressiveClassifier
clf = PassiveAggressiveClassifier(n_iter=50)
clf = clf.fit(X_train_tfidf, trainlabel)
predicted = clf.predict(X_new_tfidf)

for t in range(0, ntest):
    print predicted[t]






