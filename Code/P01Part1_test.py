#import necessary package
import re
import os
import numpy as np
import sys
from collections import Counter
from scipy import spatial
from sklearn.feature_extraction.text import CountVectorizer

stopword = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 
'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 
'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 
'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 
'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 
'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 
'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 
'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 
'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 
'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 
'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 
'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 
'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 
'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 
'further', 'was', 'here', 'than','shes','']

#Function template function voc = buildVoc(folder, voc);
#Inputs:
#a folder is a folder path, which contains training data
#b voc is a cell array to which the vocabulary is added so you can build a single lexicon for a set of
#folders, the first time you call the  fun voc is an empty cell array { }
def buildVoc(folderpath,voc):

    for filename in os.listdir(folderpath):
        if filename.endswith(".txt"):
            review = []
            filepath = folderpath + filename
            f = open(filepath, 'r')
            review.append(f.read())
            review[0] = re.sub(r'[^\w\s]', '', review[0])
            review[0] = re.sub("i'll|i've|didn't|you'|i'm|(it+s)|she's|`", '', review[0])
            review[0] = re.sub('[0-9]', '', review[0])
            #lowercase all words
            review[0] = review[0].lower()
            split_review = review[0].split()
            #get ride of stop words
            stop_result=[]
            for item in split_review:
                if item not in stopword and len(item) > 2:
                    stop_result.append(item)
            removed_review = ' '.join(map(str,stop_result))
            parse_review=removed_review.split()
            c = Counter(parse_review)
            threshold = 3
            # result=[]
            for i in c.items():
                if i[1] > threshold:
                    voc.append(i[0])
            # voc.append(result)
    return voc

train_neg_voc=buildVoc('../Data/kNN/training/neg/', [])
train_pos_voc=buildVoc('../Data/kNN/training/pos/', [])
voc = train_neg_voc + train_pos_voc
voc = list(filter(lambda x:voc.count(x) == 1, voc))

def feature_extrction(filepath, voc):
    feature = []
    review = []
    f=open(filepath, 'r')
    review.append(f.read())
    review[0] = re.sub(r'[^\w\s]', '', review[0])
    review[0] = re.sub("i'll|i've|didn't|you'|i'm|(it+s)|she's|`", '', review[0])
    review[0] = re.sub('[0-9]', '', review[0])
    #lowercase all words
    review[0] = review[0].lower()
    split_review = review[0].split()
    #get ride of stop words
    stop_result=[]
    for item in split_review:
        if item not in stopword:
            stop_result.append(item)
    removed_review = ' '.join(map(str,stop_result))
    parse_review=removed_review.split()
    # print(len(parse_review))

    feat_vec = np.zeros(len(voc))
    for item in parse_review:
        if item in voc and len(item) > 2:
            feat_vec[voc.index(item)] += 1
    # c = Counter(parse_review)
    # for i in c.items():
    #     if i[1]>2:
    #         feature.append(i[1])
    return feat_vec


#get testing txt files feature
test_neg_all = []
for filename in os.listdir('../Data/kNN/testing/neg/'):
    if filename.endswith(".txt"):
        filepath = '../Data/kNN/testing/neg/' + filename
        test_neg_feat = list(feature_extrction(filepath, voc))
        test_neg_feat.append(0)
        test_neg_all.append(test_neg_feat)
        
        
test_pos_all=[]
for filename in os.listdir('../Data/kNN/testing/pos/'):
    if filename.endswith(".txt"):
        filepath = '../Data/kNN/testing/pos/' + filename
        test_pos_feat = list(feature_extrction(filepath, voc))
        test_pos_feat.append(1)
        test_pos_all.append(test_pos_feat)


# test_feat is combiend with 10 test negative reviews feature plus 10 positive reviews feature.        
test_all = test_neg_all+test_pos_all


# get training txt files feature
train_neg_all = []
for filename in os.listdir('../Data/kNN/training/neg/'):
    if filename.endswith(".txt"):
        filepath = '../Data/kNN/training/neg/' + filename
        train_neg_feat = list(feature_extrction(filepath, voc))
        train_neg_feat.append(0)
        train_neg_all.append(train_neg_feat)

train_pos_all=[]
for filename in os.listdir('../Data/kNN/training/pos/'):
    if filename.endswith(".txt"):
        filepath = '../Data/kNN/training/pos/' + filename
        train_pos_feat = list(feature_extrction(filepath, voc))
        train_pos_feat.append(1)
        train_pos_all.append(train_pos_feat)

# test_feat is combiend with 90 train negative reviews feature plus 90 positive reviews feature.    
train_all = train_neg_all + train_pos_all

# calculate the sum squared distance between two vectors
def SSD(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return distance

def commonWords(row1, row2):
    distance = 0
    for i, j in zip(row1, row2):
        if i > 0 and j > 0:
            distance += 1
    # row3 = set(row1) & set(row2)
    # row4 = sorted(row3, key = lambda k : row1.index(k))
    if distance == 0:
        distance = sys.maxsize
    else:
        distance = 1 / distance
    return distance

def cosDistance(row1, row2):
    res = spatial.distance.correlation(row1, row2)
    return res

# Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        # USE DIFFERENT DISTANCE

        dist = commonWords(test_row, train_row)

        distances.append((train_row, dist))
    distances.sort(key = lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0]) 
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction

def acc(train_all, test_all, number_of_neighboor):
    count = 0
    result = []
    for i in range(len(test_all)):
        prediction = predict_classification(train_all, test_all[i], number_of_neighboor)
        result.append(prediction)
        if prediction==test_all[i][-1]:
            count+=1
    acc=count / len(test_all)
    return acc, result

print("")
print(acc(train_all,test_all, 14))