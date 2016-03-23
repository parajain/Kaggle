#Feature gen using word2vec

import json
from sklearn.cluster import KMeans
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier

##########################################
# Define a function to create bags of centroids
#
def create_bag_of_centroids( wordlist, word_centroid_map ):
    #
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max( word_centroid_map.values() ) + 1
    #
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
    #
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    #
    # Return the "bag of centroids"
    return bag_of_centroids

#input file in the form id tab sentence
input_file = ""

id_train = []
X_train = []
with open(input_file, 'r') as f:
	for line in f:
		lsplit = line.split('\t')
		id_train.append(lsplit[0])
		X_train.append(lsplit[1].split(' '))


print(X_train[0]) 

print('Word2vec')

# Set values for various parameters
num_features = 100    # Word vector dimensionality                      
min_word_count = 1   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 20          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
print "Training model..."
print len(X_train)

model = word2vec.Word2Vec(X_train, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context)
print 'Model word2vec completed'

# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
# average of 5 words per cluster
word_vectors = model.syn0
#num_clusters = word_vectors.shape[0] / 5
num_clusters = 200
# Initalize a k-means object and use it to extract centroids
print "Running K means"
kmeans_clustering = KMeans( n_clusters = num_clusters )
idx = kmeans_clustering.fit_predict( word_vectors )
# Create a Word / Index dictionary, mapping each vocabulary word to
# a cluster number
word_centroid_map = dict(zip( model.index2word, idx ))

# Pre-allocate an array for the training set bags of centroids (for speed)
sentence_feature = np.zeros( ( len(X_train), num_clusters),dtype="float32" )

# Transform the training set reviews into bags of centroids
counter = 0
for review in X_train:
    sentence_feature[counter] = create_bag_of_centroids( review, word_centroid_map )
    counter += 1

print len(sentence_feature[0])
print len(sentence_feature)


out_dict = {}
out_file = open('features_out.txt', 'a')
for i in range(len(id_train)):
	out_dict[id_train[i]] = sentence_feature[i]

for id in out_dict:
	out_file.write("{}\t{}\n".format(id, out_dict.get(id)))

out_file.close()

