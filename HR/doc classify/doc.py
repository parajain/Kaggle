import re
import numpy
traindata = []
trainlabel = []

splitter = re.compile('\\W*')
with open('trainingdata.txt') as f:
    idx = 0
    for line in f:
        if idx != 0:
            line = line.rstrip('\n').strip()
            line = splitter.split(line)
            trainlabel.append(line[0])
            doc = []
            for t in range(1, len(line)):
                doc.append(line[t])
            traindata.append(' '.join(doc))
        else:
            idx = idx + 1

#print traindata
#print trainlabel

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

ntest = input()
testdoc = []
for t in range(0, ntest):
    doc = raw_input()
    testdoc.append(doc)

X_new_counts = count_vect.transform(testdoc)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
""""
#Naive bayes
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, trainlabel)
predicted = clf.predict(X_new_tfidf)

#test random forest

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X_train_tfidf, trainlabel)
predicted = clf.predict(X_new_tfidf)
"""
from sklearn.linear_model import PassiveAggressiveClassifier
clf = PassiveAggressiveClassifier(n_iter=50)
clf = clf.fit(X_train_tfidf, trainlabel)
predicted = clf.predict(X_new_tfidf)

for t in range(0, ntest):
    print predicted[t]
