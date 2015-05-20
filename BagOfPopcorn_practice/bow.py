import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def review_to_words( raw_review ):
    #remove html
    rev_text = BeautifulSoup(raw_review).get_text()
    letters = re.sub("[^A-Za-z]", " ", rev_text)
    words = letters.lower().split()
    #create a set of stopwords(from nltk) list, set is faster to search 
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return (" ".join(meaningful_words))



train = pd.read_csv("data/labeledTrainData.tsv",header=0,delimiter="\t",quoting=3)
#quoting=3 tells Python to ignore doubled quote

num_reviews = train["review"].size


clean_train_reviews = []

for i in xrange(0, num_reviews):
    if( (i+1)%1000 == 0 ):
        print "Review %d of %d"  % ( i+1, num_reviews)
    clean_train_reviews.append( review_to_words( train["review"][i] ))

print "Creating the bag of words...\n"
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(clean_train_reviews) #?????????????????????????????????????????

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()

print train_data_features.shape

# Take a look at the words in the vocabulary
vocab = vectorizer.get_feature_names()
#print vocab

######### RANDOM FOREST
print "Training the random forest..."
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( train_data_features, train["sentiment"] )

######TEST##########



test = pd.read_csv("data/testData.tsv",header=0,delimiter="\t",quoting=3)

print test.shape

num_reviews = len(test["review"])
clean_test_reviews = []


for i in xrange(0, num_reviews):
    if( (i+1)%1000 == 0):
        print i
    clean_test_reviews.append( review_to_words( test["review"][i] ))

test_data_features = vectorizer.transform(clean_test_reviews) #only transform here
test_data_features = test_data_features.toarray()


result = forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

# Use pandas to write the comma-separated output file
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )














