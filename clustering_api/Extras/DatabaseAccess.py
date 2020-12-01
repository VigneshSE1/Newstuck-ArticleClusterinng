import mysql.connector
import datetime
from datetime import datetime, timedelta
import json

#mydb = mysql.connector.connect(host="52.188.110.40",port=3307,user="user",password="tWXg5p8FK6JpvICDcYQ%fppxbJa",database="newstuckstage")
mydb = mysql.connector.connect(host="127.0.0.1",port=3306,user="root",password="me@1nd1a",database="MobileNewsTuckTwo")
mycursor = mydb.cursor(buffered=True)
mycursor = mydb.cursor(prepared=True)
mycursor = mydb.cursor()

def getTodayNewsFromDb():
    today = datetime.now()
    yesterday = datetime.today() - timedelta(days=1)

    today = today.strftime("%Y-%m-%d 18:30:00")
    yesterday = yesterday.strftime("%Y-%m-%d 18:30:00")

    query = """SELECT FeedItemId,Title,Language FROM FeedItems WHERE PublishDate >= (%s) and PublishDate <= (%s)"""
    dateValues = (yesterday,today)

    mycursor.execute(query,dateValues)

    row_headers=[x[0] for x in mycursor.description] #this will extract row headers
    resultsFromDatabase = mycursor.fetchall()

    json_data=[]

    for result in resultsFromDatabase:
        json_data.append(dict(zip(row_headers,result)))

    #mycursor.close()
    #mydb.close()
    return json_data
#with open('DBResults.json', 'w', encoding='utf-8') as f:
    #json.dump(json_data, f, ensure_ascii=False, indent=4)

def commitResultToDataBase(resultJson):
    #print(resultJson)
    Updatequery = """UPDATE FeedItems SET ClusterId = (%s), Language = (%s)  WHERE FeedItemId = (%s)"""
    for news in resultJson:
       # print(news["ClusterId"],news["Language"],news["FeedItemId"])
        values = (news["ClusterId"],news["Language"],news["FeedItemId"])
        mycursor.execute(Updatequery,values)
        mydb.commit()
    no_of_commits = mycursor.rowcount
    mycursor.close()
    mydb.close()

    return no_of_commits