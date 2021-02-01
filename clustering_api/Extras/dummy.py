# Write the ClusteredOutput into JSON File
# def writeClsuterdJson(clusteredJson,filename):
#     with open(filename, 'w', encoding='utf-8') as f:
#         json.dump(clusteredJson, f, ensure_ascii=False, indent=4)


# __Main__
#newsArticleJson = getNewsArticlesJson()
#seperateBylanguage = seperateArticlesByLanguage(newsArticleJson)
#print(seperateBylanguage)


# import flask
# from flask import request

# app = flask.Flask(__name__)
# #app.config["DEBUG"] = False

# @app.route('/getCluster/', methods=["POST"])
# def Clustering():

#     try:
#         language = request.form['language']
#         jsonTitles = request.form['titles']
#         newsTitles = getNewsTitlesFromJson(jsonTitles)
#         newsVectors = getVectorsFromFastText(newsTitles,language)
#         noOfClusters = findSilhouetteMaxScore(newsVectors)
#         clusteredJson = clusterArticleByKMeans(noOfClusters,newsVectors,jsonTitles)

#         return clusteredJson.json

#     except Exception as e:
#         return  e

# app.run()

# For TamilArticles
# newsTitlesTamil = getNewsTitlesFromJson()
# newsVectorsTamil = getVectorsFromFastText(newsTitlesTamil,"ta")
# noOfClustersTamil = findSilhouetteMaxScore(newsVectorsTamil)
# clusteredJsonTamil = clusterArticleByKMeans(noOfClustersTamil,newsVectorsTamil,seperateBylanguage["tamil"])
#writeClsuterdJson(clusteredJsonTamil,"Tamil.json")

# For EnglishArticles
# newsTitlesEnglish = getNewsTitlesFromJson(seperateBylanguage["english"])
# newsVectorsEnglish = getVectorsFromFastText(newsTitlesEnglish,"en")
# noOfClustersEnglish = findSilhouetteMaxScore(newsVectorsEnglish)
# clusteredJsonEnglish = clusterArticleByKMeans(noOfClustersEnglish,newsVectorsEnglish,seperateBylanguage["english"])
#writeClsuterdJson(clusteredJsonEnglish,"English.json")

#Write to DataBase
#clusteredJsonTamil.extend(clusteredJsonEnglish)

#from DatabaseAccess import commitResultToDataBase

#rowsAffected = commitResultToDataBase(clusteredJsonTamil)
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
