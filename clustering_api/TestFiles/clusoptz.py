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

# new imports
import pandas as pd
from datetime import datetime, timedelta
import dateutil

# from waitress import serve
print('loading ta model')
ta_model = fasttext.load_model("cc.ta.100.bin")
print('loading en model')
en_model = fasttext.load_model("cc.en.100.bin")


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
        print('ta')

        model = ta_model
        print("Tamil Model Loaded")
       
    elif(language == "en"):
        print('en')
        model = en_model
        print("English Model Loaded")

    for title in titleList:
       # print(title)
        a = model.get_sentence_vector(title)
        # print(a)
        vectorValues.append(a)

    numpyVectorArray = np.array(vectorValues)
    print("NumpyArray Generated")
    return numpyVectorArray

# Find the Optimal Number Of Cluster using SilhouetteMaxScore Method
def findSilhouetteMaxScore(vectorArray):
    print("inside findSilhouetteMaxScore")
    length = len(vectorArray)
    if length == 1:
        return 1
    elif length < 10:
        # start = 2
        # end = length
        return length
        
    elif length >= 10:
        # start = length//3
        # end = length - start
        print(length)
        print(length//2)
        return length//2

    # silhouetteScore = []

    # for n_clusters in range((int)(start),(int)(end)): 
    #     cluster = KMeans(n_clusters = n_clusters) 
    #     cluster_labels = cluster.fit_predict(vectorArray)
    #     silhouette_avg = silhouette_score(vectorArray, cluster_labels)
    #     silhouetteScore.append(silhouette_avg)
    # #print(silhouetteScore)
    # maxpos = silhouetteScore.index(max(silhouetteScore))
    # print("SilhouetteMaxScore found")
    # return maxpos+start

# Cluster the NewsArticle BY K-Means
def clusterArticleByKMeans(clusterNumber,vectors,newsArticleJson):
    print("inside clusterArticleByKMeans")
    clf = KMeans(n_clusters = clusterNumber, init = 'k-means++')
    labels = clf.fit_predict(vectors)

    for index, newsArticle in enumerate(newsArticleJson):
        labelValue = labels[index] 
        newsArticle["ClusterId"] = int(labelValue)+1
    print("cluster by kmeans done")
    return sorted(newsArticleJson, key = lambda i: (i['ClusterId']))

def detectLanguage(datas):
    # print("detect Language Function Called")
    for x in datas:
            print("inside For")
            language = detect(x["Title"])
            x["Language"] = language
    # print(datas)
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

def incrementalSilhouetteMaxScore(vectors_old, vectors_new, existing_k = None):
    if len(vectors_old) == 0:
        vectorArray = vectors_new
    elif len(vectors_new) == 0:
        vectorArray = vectors_old
    else: 
        vectorArray = vectors_old + vectors_new
    
    length = len(vectorArray)
    
    if existing_k:
        start = existing_k
        end = existing_k + len(vectors_new)
    else:
        print('K value not found in config.json')
        if length == 1:
            return 1
        elif length < 10:
            start = 2
            end = length
            # return length    
        elif length >= 10:
            start = length//5
            end = max(length//3, 10)
            # print(length)
            # print(length//2)
            # return length//2
        print(f'Initializing range ({str(start)}, {str(end)}) for Silhoutte method.') 

    silhouetteScore = []
    
    for n_clusters in range((int)(start),(int)(end)): 
        cluster = KMeans(n_clusters = n_clusters) 
        cluster_labels = cluster.fit_predict(vectorArray)
        silhouette_avg = silhouette_score(vectorArray, cluster_labels)
        silhouetteScore.append(silhouette_avg)
    
    if silhouetteScore:
        maxpos = silhouetteScore.index(max(silhouetteScore))
        print("SilhouetteMaxScore found")
        return maxpos+start, vectorArray
    else:
        return existing_k, vectorArray
        
def get_k_value(current_date, language):
    try:
        with open('config.json') as f:
            config = json.load(f)
        return config[language][current_date] 
    except (json.JSONDecodeError, FileNotFoundError):
        print('Incorrect format in config file, initializing again....')
        with open('config.json', 'w') as f:
            json.dump({}, f)
        return None
    except:
        return None

def put_k_value(noOfClusters, current_date, language):
    try:
        with open('config.json') as f:
            config = json.load(f)
        if language in config:
            config[language][current_date] = noOfClusters
        else:
            config[language] = {
                current_date: noOfClusters
            }
        with open('config.json', 'w') as f:
            json.dump(config, f)
    except (json.JSONDecodeError, FileNotFoundError):
        print('Incorrect format in config file, initializing again....')
        with open('config.json', 'w') as f:
            to_insert = {
                language: {
                current_date: noOfClusters  
                }
            }
            json.dump(to_insert, f)
    except Exception as e:
        print(e)


# Api endPoint
import flask
from flask import request, jsonify, Response
# from werkzeug.contrib.fixers import ProxyFix

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/api/v1/getcluster', methods=['POST'])
def cluster_all():
    print("get Cluster Api Called")
    req_data = request.get_json()
    language = req_data['Language']
    #print("->>>>>>>>>>>>>>>>>>>>" , language)
    jsonTitles = req_data['Titles']

    # Seggregating data according to timestamp
    title_df = pd.DataFrame(jsonTitles)
    print(title_df)
    title_df['PublishDate'] = title_df['PublishDate'].apply(
        lambda x: dateutil.parser.parse(x))
    print("________________________________________________________________________________________________")
    print(title_df['PublishDate'])
    # put 20 minutes considering scrapper latency=0, needs to be adjusted accordingly
    current_utc = datetime.utcnow() 
    adjusted_utc = current_utc - timedelta(hours=0, minutes= 20)
    current_date = current_utc.date()
    time_mask = title_df['PublishDate'] < adjusted_utc
    print("__________________TIME MASK_______________________")
    print(time_mask)
    old_titles = list(title_df[time_mask].Title)
    print("_____________________OLD TITLES_____________________________")
    print(old_titles)
    new_titles = list(title_df[~time_mask].Title)
    print("_____________________NEW TITLES_____________________________")
    print(new_titles)



    #print("->>>>>>>>>>>>>>>>>>>>" , jsonTitles)
    # old_titles = getNewsTitlesFromJson(old_titles)
    # new_titles = getNewsTitlesFromJson(new_titles)

    old_vectors = getVectorsFromFastText(old_titles, language)
    new_vectors = getVectorsFromFastText(new_titles, language)

    existing_k = get_k_value(str(current_date), language)
    #findElbowFromVector(newsVectors)
    # noOfClusters = findSilhouetteMaxScore(newsVectors)
    noOfClusters, newsVectors = incrementalSilhouetteMaxScore(old_vectors, new_vectors, existing_k)
    put_k_value(noOfClusters, str(current_date), language)

    clusteredJson = clusterArticleByKMeans(noOfClusters,newsVectors,jsonTitles)
    clusteredJsonResult = json.dumps(clusteredJson,ensure_ascii=False,indent=4)
    return clusteredJsonResult

@app.route('/api/v1/detectlanguage', methods=['POST'])
def lanuageDetect_all():
    print("LanguageDetection Api Called")
    req_data = request.get_json()
    #print(req_data)
    result_data = detectLanguage(req_data)
    return  jsonify(result_data)

def run():
    # global global_fn_main
    # global_fn_main = fn_main
    # app.run(debug = False, port = 8080, host = '0.0.0.0', threaded = True)
# app.wsgi_app = ProxyFix(app.wsgi_app)
# if __name__ == '__main__':
    # app.run(debug = False, port = 80, host = '0.0.0.0', threaded = True)
#     app.run(host='0.0.0.0', port=80)
    app.run()
    # serve(app, host='127.0.0.1', port=5000)

run()