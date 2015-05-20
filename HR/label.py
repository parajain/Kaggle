#!/usr/bin/python
from sys import stdin
import re
import string

stoplist = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours  ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']

def removestop(stoplist,inplist):
    for item in stoplist:
        if item in inplist:
            inplist.remove(item)
    return inplist


line = stdin.readline().strip().split()
wordlist = {}
labelset = set()
t = int(line[0])
e = int(line[1])
test = []
que = []
labels = []
print t
for i in range(t):
    x = []
    st = stdin.readline().strip() 
    x = st.split()
#    print st
    labels.append(x[1:])
    q = stdin.readline().strip().strip('?').lower().split()
    q = removestop(stoplist,q)
    que.append(q)
    for w in q:
        if w not in wordlist.keys():
            s = set(x[1:])
            wordlist[w] = s
        else:
            s = set(x[1:])
            s2 = wordlist[w]
            s = s & s2
            wordlist[w] = s

for key,value in wordlist.iteritems():
    print key,value
#print "============"
#print labels
#print que
#print labels

for i in range(e):
    x = []
    x = stdin.readline().strip().strip('?').lower().split()
    x = removestop(stoplist,x)
    test.append(x)
    s = set()
    for w in x:
       if w in wordlist.keys():
           s = s | wordlist[w]
    print s   
print test






