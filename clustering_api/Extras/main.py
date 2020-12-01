import numpy as np
import nltk
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs 
from sklearn.metrics import silhouette_score 
from sklearn import metrics
from sklearn import cluster
from sklearn.decomposition import PCA
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist
import json
import fasttext
import fasttext.util
import math
import pickle

import DatabaseAccess
from DatabaseAccess import getTodayNewsFromDb

# Read DataSet and Return the JSON Data
def getNewsArticlesJson():
    newsArticles = getTodayNewsFromDb()
    #print(newsArticles)
    return newsArticles

import LanguageDetection
from LanguageDetection import detectLanguage

def seperateArticlesByLanguage(newsArticles):
    language_seperated = detectLanguage(newsArticles)
    return language_seperated

# Make Array Of Title's
def getNewsTitlesFromJson(jsonData):
    ArrayOfSentence = []
    for data in jsonData:
        splittedSentence = data["Title"]
        ArrayOfSentence.append(splittedSentence)
    return ArrayOfSentence

# Generate VectorForms as NumPy Array Using FastText Model
def getVectorsFromFastText(titleList,language):
    vectorValues = []
    if(language == "ta"):
        model = fasttext.load_model("D:/Models/cc.ta.300.bin")
    elif(language == "en"):
        model = fasttext.load_model("D:/Models/cc.en.300.bin")

    for title in titleList:
        a = model.get_word_vector(title)
        vectorValues.append(a)

    numpyVectorArray = np.array(vectorValues)
    return numpyVectorArray

# Find the Optimal Number Of Cluster using SilhouetteMaxScore Method
def findSilhouetteMaxScore(vectorArray):
    silhouetteScore = []
    for n_clusters in range(2,len(vectorArray)): 
        cluster = KMeans(n_clusters = n_clusters) 
        cluster_labels = cluster.fit_predict(vectorArray)
        silhouette_avg = silhouette_score(vectorArray, cluster_labels)
        silhouetteScore.append(silhouette_avg)
   # print(silhouetteScore)
    maxpos = silhouetteScore.index(max(silhouetteScore)) 
   # print(maxpos+2)
    return maxpos+2
 
# Cluster the NewsArticle BY K-Means
def clusterArticleByKMeans(clusterNumber,vectors,newsArticleJson):
    clf = KMeans(n_clusters = clusterNumber, init = 'k-means++')
    labels = clf.fit_predict(vectors)

    for index, newsArticle in enumerate(newsArticleJson):
        labelValue = labels[index] 
        newsArticle["ClusterId"] = int(labelValue)+1
    return sorted(newsArticleJson, key = lambda i: (i['ClusterId']))

# Write the ClusteredOutput into JSON File
def writeClsuterdJson(clusteredJson,filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(clusteredJson, f, ensure_ascii=False, indent=4)


# __Main__
newsArticleJson = getNewsArticlesJson()
seperateBylanguage = seperateArticlesByLanguage(newsArticleJson)
#print(seperateBylanguage)

# For TamilArticles
newsTitlesTamil = getNewsTitlesFromJson(seperateBylanguage["tamil"])
newsVectorsTamil = getVectorsFromFastText(newsTitlesTamil,"ta")
noOfClustersTamil = findSilhouetteMaxScore(newsVectorsTamil)
clusteredJsonTamil = clusterArticleByKMeans(noOfClustersTamil,newsVectorsTamil,seperateBylanguage["tamil"])
#writeClsuterdJson(clusteredJsonTamil,"Tamil.json")

# For EnglishArticles
newsTitlesEnglish = getNewsTitlesFromJson(seperateBylanguage["english"])
newsVectorsEnglish = getVectorsFromFastText(newsTitlesEnglish,"en")
noOfClustersEnglish = findSilhouetteMaxScore(newsVectorsEnglish)
clusteredJsonEnglish = clusterArticleByKMeans(noOfClustersEnglish,newsVectorsEnglish,seperateBylanguage["english"])
#writeClsuterdJson(clusteredJsonEnglish,"English.json")

#Write to DataBase
clusteredJsonTamil.extend(clusteredJsonEnglish)

from DatabaseAccess import commitResultToDataBase

rowsAffected = commitResultToDataBase(clusteredJsonTamil)
#----------------------------------------------------------------------------------------------------------------------------------------S

# Find the Optimal Number Of Cluster using Elbow Method
# def findElbowFromVector(vectorArray):
#     elbow=[]
#     for i in range(1, 10):
#         kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 101)
#         kmeans.fit(vectorArray)
#         elbow.append(kmeans.inertia_)
#     elbowBend = 5
#     return elbowBend

# def calculate_wcss(data):
#     wcss = []
#     for n in range(2, len(data)):
#         kmeans = KMeans(n_clusters=n)
#         kmeans.fit(X=data)
#         wcss.append(kmeans.inertia_)
    
#     #print(wcss)
#     return wcss

# def optimal_number_of_clusters(wcss):
#     x1, y1 = 2, wcss[0]
#     x2, y2 = 20, wcss[len(wcss)-1]

#     distances = []
#     for i in range(len(wcss)):
#         x0 = i+2
#         y0 = wcss[i]
#         numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
#         denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
#         distances.append(numerator/denominator)
    
#     print(distances.index(max(distances)) + 2)
#     return distances.index(max(distances)) + 2

#noOfClusters = findElbowFromVector(newsVectors)
#wcss = calculate_wcss(newsVectors)
#onc = optimal_number_of_clusters(wcss)