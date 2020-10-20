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
from gensim.models.wrappers import FastText
from datetime import datetime

from langdetect import detect


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
       # print(title)
        a = model.get_sentence_vector(title)
        #print(a)
        vectorValues.append(a)

    numpyVectorArray = np.array(vectorValues)
    return numpyVectorArray

# Find the Optimal Number Of Cluster using SilhouetteMaxScore Method
def findSilhouetteMaxScore(vectorArray):
    length = len(vectorArray)
    if length == 1:
        return 1
    elif length < 10:
        start = 2
        end = length
        
    elif length >= 10:
        start = length//3
        end = length - start

    silhouetteScore = []
    for n_clusters in range((int)(start),(int)(end)): 
        cluster = KMeans(n_clusters = n_clusters) 
        cluster_labels = cluster.fit_predict(vectorArray)
        silhouette_avg = silhouette_score(vectorArray, cluster_labels)
        silhouetteScore.append(silhouette_avg)
   # print(silhouetteScore)
    maxpos = silhouetteScore.index(max(silhouetteScore)) 
    #print(maxpos+2)
    return maxpos

# Cluster the NewsArticle BY K-Means
def clusterArticleByKMeans(clusterNumber,vectors,newsArticleJson):

    clf = KMeans(n_clusters = clusterNumber, init = 'k-means++')
    labels = clf.fit_predict(vectors)

    for index, newsArticle in enumerate(newsArticleJson):
        labelValue = labels[index] 
        newsArticle["ClusterId"] = int(labelValue)+1
    return sorted(newsArticleJson, key = lambda i: (i['ClusterId']))

def detectLanguage(datas):
    for x in datas:
            language = detect(x["Title"])
            x["Language"] = language
    return datas

# def findElbowFromVector(vectorArray):
#     # elbow=[]
#     # for i in range(1, len(vectorArray)):
#     #     kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 101)
#     #     kmeans.fit(vectorArray)
#     #     elbow.append(kmeans.inertia_)
#     now = datetime.now()

#     current_time = now.strftime("%H:%M:%S")
#     print("Current Time =", current_time)

#     distortions = [] 
#     inertias = [] 
#     mapping1 = {} 
#     mapping2 = {} 
#     K = range(1,len(vectorArray)) 
  
#     for k in K: 
#         #Building and fitting the model 
#         kmeanModel = KMeans(n_clusters=k).fit(vectorArray) 
#         kmeanModel.fit(vectorArray)     
      
#         distortions.append(sum(np.min(cdist(vectorArray, kmeanModel.cluster_centers_, 
#                       'euclidean'),axis=1)) / vectorArray.shape[0]) 
#         inertias.append(kmeanModel.inertia_) 
  
#         mapping1[k] = sum(np.min(cdist(vectorArray, kmeanModel.cluster_centers_, 
#                  'euclidean'),axis=1)) / vectorArray.shape[0] 
#         mapping2[k] = kmeanModel.inertia_
#     now = datetime.now()

#     current_time = now.strftime("%H:%M:%S")
#     print("Current Time =", current_time)
#     plt.plot(K, distortions, 'bx-') 
#     plt.xlabel('Values of K') 
#     plt.ylabel('Distortion') 
#     plt.title('The Elbow Method using Distortion') 
#     plt.show()

# Api endPoint
import flask
from flask import request, jsonify, Response

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/api/v1/getcluster', methods=['POST'])
def cluster_all():
    req_data = request.get_json()
    language = req_data['Language']
    #print("->>>>>>>>>>>>>>>>>>>>" , language)
    jsonTitles = req_data['Titles']
    #print("->>>>>>>>>>>>>>>>>>>>" , jsonTitles)
    newsTitles = getNewsTitlesFromJson(jsonTitles)
    newsVectors = getVectorsFromFastText(newsTitles,language)
    #findElbowFromVector(newsVectors)
    noOfClusters = findSilhouetteMaxScore(newsVectors)
    clusteredJson = clusterArticleByKMeans(noOfClusters,newsVectors,jsonTitles)
    clusteredJsonResult = json.dumps(clusteredJson,ensure_ascii=False,indent=4)
    return clusteredJsonResult

@app.route('/api/v1/detectlanguage', methods=['POST'])
def lanuageDetect_all():
    req_data = request.get_json()
    #print(req_data)
    result_data = detectLanguage(req_data)
    return  jsonify(result_data)

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=80)
    app.run()