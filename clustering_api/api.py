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
import logging
from langdetect import detect
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.wrappers import FastText
from urllib.request import urlopen
import sys
import os

# new imports
import pandas as pd
from datetime import datetime, timedelta
import dateutil
import redis
import os
import logging

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig( format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='[%Y-%m-%d %H:%M:%S +0000]')
# from waitress import serve
model_lang = None
model = None

punctuation_signs = list("?:!.,;")
stop_words = list(stopwords.words('english'))

# Make Array Of Title's
def getNewsTitlesFromJson(jsonData):
    ArrayOfSentence = []
    for data in jsonData:
        splittedSentence = clean_text(data["Title"])
        ArrayOfSentence.append(splittedSentence)
    logging.info('Summaries Generated From Json')
    return ArrayOfSentence

def clean_text(text):
    if text:
        text = text.replace("\r", " ")
        text = text.replace("\n", " ")
        text = text.replace("    ", " ")
        text = text.replace('"', '')
        text = text.lower()
        for punct_sign in punctuation_signs:
            text = text.replace(punct_sign, '')
        wordnet_lemmatizer = WordNetLemmatizer()
        text_list = []
        for word in text.split(' '):
            text_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
        text = ' '.join(text_list)

        for stop_word in stop_words:
            regex_stopword = r"\b" + stop_word + r"\b"
            text = text.replace(regex_stopword, '')
        return text
    return ''

# Generate VectorForms as NumPy Array Using FastText Model
def getVectorsFromFastText(titleList,language):
    global model
    global model_lang
    vectorValues = []

    if model is None or model_lang != language:
        model = None
        if(language == "ta") and model_lang != language:
            
            logging.info('Tamil Model Loading...')
            #model = fasttext.load_model("https://stagevoterliststrg.blob.core.windows.net/newstuck-cluster-model/ta/cc.ta.100.bin")
            # fasttext.util.download_model('ta', if_exists='ignore',dimension=100)
            exists = os.path.isfile('cc.ta.300.bin')
            if exists:
                 logging.info('Tamil Model Exists')
            else:   
                logging.info('Tamil Model Downloading...')
                # _download_file("https://stagevoterliststrg.blob.core.windows.net/newstuck-cluster-model/ta/cc.ta.100.bin","cc.ta.100.bin")
                fasttext.util.download_model('ta', if_exists='ignore')       
            # f = open("cc.ta.100.bin", 'r',encoding='utf-8')
            model = fasttext.load_model("cc.ta.300.bin")
            # model = KeyedVectors.load_word2vec_format()
            logging.info('Tamil Model Load Completed')
        
        elif(language == "en") and model_lang != language:
            logging.info('English Model Loading...')
            exists = os.path.isfile('cc.en.300.bin')
            if exists:
                logging.info('English Model Exists')
            else:
                logging.info('English Model Downloading...')
                # _download_file("https://stagevoterliststrg.blob.core.windows.net/newstuck-cluster-model/en/cc.en.100.bin","cc.en.100.bin")
                fasttext.util.download_model('en', if_exists='ignore')
            # fasttext.util.download_model('en', if_exists='ignore',dimension=100) 
            # model = fasttext.load_model("https://stagevoterliststrg.blob.core.windows.net/newstuck-cluster-model/en/cc.en.100.bin")
            # f = open("https://stagevoterliststrg.blob.core.windows.net/newstuck-cluster-model/en/cc.en.100.bin", 'r')
            model = fasttext.load_model("cc.en.300.bin")
            # model = fasttext.load_model("cc.en.100.bin")
            logging.info('English Model LoadComplete')

    model_lang = language
    for title in titleList:
        # print(title)
        a = model.get_sentence_vector(title.replace('\n',""))
        # print(a)
        vectorValues.append(a)

    numpyVectorArray = np.array(vectorValues)
    logging.info('Vector Values Numpy Array Generated')
    return numpyVectorArray

# Find the Optimal Number Of Cluster using SilhouetteMaxScore Method
def findSilhouetteMaxScore(vectorArray):
    logging.info('Finding Max Score Started')
    length = len(vectorArray)
    if length == 1:
        logging.info('Got Optimal Cluster Value')
        return 1
    elif length < 10:
        # start = 2
        # end = length
        logging.info('Got Optimal Cluster Value')
        return length
        
    elif length >= 10:
        # start = length//3
        # end = length - start
        # print(length)
        # print(length//2)
        logging.info('Got Optimal Cluster Value')
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
    logging.info('Cluster By K-Means Started')
    clf = KMeans(n_clusters = clusterNumber, init = 'k-means++')
    labels = clf.fit_predict(vectors)

    for index, newsArticle in enumerate(newsArticleJson):
        labelValue = labels[index] 
        newsArticle["ClusterId"] = int(labelValue)+1
    logging.info('Cluster By K-Means Commpleted')
    return sorted(newsArticleJson, key = lambda i: (i['ClusterId']))

def detectLanguage(datas):
    logging.info('Language Detection Started')
    for x in datas:
            # print("inside For")
            language = detect(x["Title"])
            x["Language"] = language
    logging.info('Language Detection Completed')
    return datas

def _download_file(url, write_file_name, chunk_size=2**13):
    print("Downloading %s" % url)
    response = urlopen(url)
    if hasattr(response, 'getheader'):
        file_size = int(response.getheader('Content-Length').strip())
    else:
        file_size = int(response.info().getheader('Content-Length').strip())
    downloaded = 0
    download_file_name = write_file_name + ".part"
    with open(download_file_name, 'wb') as f:
        while True:
            chunk = response.read(chunk_size)
            downloaded += len(chunk)
            if not chunk:
                break
            f.write(chunk)
            _print_progress(downloaded, file_size)

    os.rename(download_file_name, write_file_name)

def _print_progress(downloaded_bytes, total_size):
    percent = float(downloaded_bytes) / total_size
    bar_size = 50
    bar = int(percent * bar_size)
    percent = round(percent * 100, 2)
    sys.stdout.write(" (%0.2f%%) [" % percent)
    sys.stdout.write("=" * bar)
    sys.stdout.write(">")
    sys.stdout.write(" " * (bar_size - bar))
    sys.stdout.write("]\r")
    sys.stdout.flush()

    if downloaded_bytes >= total_size:
        sys.stdout.write('\n')

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

# Incremental Clustering
def incrementalSilhouetteMaxScore(vectorArray, prev_count, existing_noOfClusters = None):

    length = len(vectorArray)
    
    if existing_noOfClusters:
        start = existing_noOfClusters
        end = existing_noOfClusters + max(length - prev_count, 0)
    else:
        logging.info('K value not found in redis-server')
        if length == 1:
            return 1
        elif length < 10:
            start = 2
            end = length
            # return length    
        elif length >= 10:
            start = length//5
            end = max(length//3, 10)
            # logging.info(length)
            # logging.info(length//2)
            # return length//2
        logging.info(f'Initializing range ({str(start)}, {str(end)}) for Silhoutte method.') 

    silhouetteScore = []
    
    for n_clusters in range((int)(start),(int)(end)): 
        cluster = KMeans(n_clusters = n_clusters) 
        cluster_labels = cluster.fit_predict(vectorArray)
        silhouette_avg = silhouette_score(vectorArray, cluster_labels)
        silhouetteScore.append(silhouette_avg)
    
    if silhouetteScore:
        maxpos = silhouetteScore.index(max(silhouetteScore))
        logging.info("SilhouetteMaxScore found")
        return maxpos+start, vectorArray
    else:
        return existing_noOfClusters, vectorArray
        
def get_k_value(redis_client, redis_key, language):
    try:
        config = redis_client.get(redis_key)
        logging.info("Config from redis :")
        logging.info(config)
        if config:
            config = json.loads(config.decode('utf-8'))
            return config[language]['no_of_clusters'], config[language]['prev_count'] 
        else:
            return config
    except Exception as e:
        logging.info('Error while getting config')
        logging.info(str(e))
        return None

def put_k_value(redis_client, noOfClusters, length, redis_key, language):
    try:
        config = redis_client.get(redis_key)
        if config:
            config = json.loads(config.decode('utf-8'))
            temp = {
                language:
                    {   
                    'no_of_clusters': noOfClusters,
                    'prev_count': length
                }}
            config = {**config, **temp}
            redis_client.set(redis_key, json.dumps(config))
        else:
            config = {language:
                {
                    'no_of_clusters': noOfClusters,
                    'prev_count': length
                }}
            redis_client.set(redis_key, json.dumps(config))
    except Exception as e:
        logging.info('Error while putting config')
        logging.info(e)


# Api endPoint
import flask
from flask import request, jsonify, Response
# from werkzeug.contrib.fixers import ProxyFix

app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/api/v1/getcluster', methods=['POST'])
def cluster_all():
    logging.info("get Cluster Api Called")
    req_data = request.get_json()
    language = req_data['Language']
    #logging.info("->>>>>>>>>>>>>>>>>>>>" , language)
    jsonTitles = req_data['Titles']
    unique_id = req_data['UUID']

    if not jsonTitles:
        return "Error - No titles received."
    vectorArray = getVectorsFromFastText(getNewsTitlesFromJson(jsonTitles), language)

    current_utc = datetime.utcnow() 
    current_date = current_utc.date()

    redis_key = f"{unique_id}_"+str(current_date)
    logging.info(f"redis_key :")
    logging.info(redis_key)
    
    # Local config for redis
    # redis_client = redis.Redis('localhost')

    # ACI config for redis
    redis_server = os.environ['REDIS']
    redis_client = redis.Redis(redis_server)
    
    # Loading existing values in the redis server
    output = get_k_value(redis_client, redis_key, language)
    if output:
        existing_noOfClusters, prev_count = output
    else:
        existing_noOfClusters, prev_count = None, 0
    
    logging.info("existing_noOfClusters")
    logging.info(existing_noOfClusters)
    logging.info("prev_count")
    logging.info(prev_count)
    
    #findElbowFromVector(newsVectors)
    # noOfClusters = findSilhouetteMaxScore(newsVectors)
    noOfClusters, newsVectors = incrementalSilhouetteMaxScore(vectorArray, prev_count, existing_noOfClusters)
    clusteredJson = clusterArticleByKMeans(noOfClusters,newsVectors,jsonTitles)
    put_k_value(redis_client, noOfClusters, len(vectorArray), redis_key, language)
    clusteredJsonResult = json.dumps(clusteredJson,ensure_ascii=False,indent=4)
    return clusteredJsonResult

@app.route('/api/v1/detectlanguage', methods=['POST'])
def lanuageDetect_all():
    logging.info('Language Detection API Called')
    req_data = request.get_json()
    #print(req_data)
    result_data = detectLanguage(req_data)
    logging.info('Language Detection Result Sent')
    return  jsonify(result_data)

def run():
    # global global_fn_main
    # global_fn_main = fn_main
    # app.run(debug = False, port = 8080, host = '0.0.0.0', threaded = True)
# app.wsgi_app = ProxyFix(app.wsgi_app)
# if __name__ == '__main__':
    logging.info('App Started Running')
    # app.run(debug = False, port = 80, host = '0.0.0.0', threaded = True)
#     app.run(host='0.0.0.0', port=80)
    app.run()
    # serve(app, host='127.0.0.1', port=5000)