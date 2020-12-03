import logging
import requests

logging.basicConfig( format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='[%Y-%m-%d %H:%M:%S +0000]')

# Make Array Of Title's
def getNewsTitlesFromJson(jsonData):
    ArrayOfSentence = []
    for data in jsonData:
        splittedSentence = data["Title"]
        ArrayOfSentence.append(splittedSentence)
    logging.info('NewsTitles Generated From Json')
    return ArrayOfSentence

def get_summary(url):
    response = requests.get('http://52.188.110.40:8090/extract', params = {'url': url})
    if response.status_code == 200:
        summary_dict = response.json()
        return summary_dict['text']
    else:
        return ''