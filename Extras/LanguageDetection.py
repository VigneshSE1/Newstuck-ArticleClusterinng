from langdetect import detect
import json
datas =[]

def getNewsArticlesJson():
    file  = open("D:/NewsTuck_Application/Newstuck-ArticleClusterinng/DataSets/04-11-2020.json",encoding="utf8")
    datas = json.load(file)
    return datas

def writeClsuterdJson(datas):
    #print(datas)
    tamilArticles = []
    englishAticles = []
    for data in datas:
        if(data["language"] == "ta"):
            tamilArticles.append(data)
        if(data["language"] == "en"):
            englishAticles.append(data)

    with open('D:/NewsTuck_Application/Newstuck-ArticleClusterinng/DataSets/04-11-2020Tamil.json', 'w', encoding='utf-8') as f:
        json.dump(tamilArticles, f, ensure_ascii=False, indent=4)
    with open('D:/NewsTuck_Application/Newstuck-ArticleClusterinng/DataSets/04-11-2020English.json', 'w', encoding='utf-8') as f:
        json.dump(englishAticles, f, ensure_ascii=False, indent=4)

datas = getNewsArticlesJson()
for x in datas:
    language = detect(x["Title"])
    x["language"] = language
    
writeClsuterdJson(datas)


def detectLanguage(datas):
    tamilArticles = []
    englishArticles = []

    for x in datas:
        if(x["Language"] != "ta" or x["Language"] != "en"):
            language = detect(x["Title"])
            x["Language"] = language

    for data in datas:
        if(data["Language"] == "ta"):
            tamilArticles.append(data)
        if(data["Language"] == "en"):
            englishArticles.append(data)

    result = {}
    result['tamil'] = tamilArticles
    result['english'] = englishArticles

    return result

    