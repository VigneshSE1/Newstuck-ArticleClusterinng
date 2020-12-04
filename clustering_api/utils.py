import logging
import requests

logging.basicConfig( format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='[%Y-%m-%d %H:%M:%S +0000]')

# Codes will change as when the model is updated.
category_codes = {
    0: 'Business',
    1: 'Entertainment',
    2: 'Politics',
    3: 'Sports',
    4: 'Technology',
    5: 'Others'
    }

category_names = [v for k,v in category_codes.items()]

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

def get_titles_categorywise(titles_list):
    title_category_dict = {}
    logging.info(f'Received titles_list in get_titles_categorywise()')
    for _, article_data in enumerate(titles_list):
        categories = article_data['category']
        print(categories)
        for category in categories:
            if category in category_names:
                if category in title_category_dict.keys():
                    title_category_dict[category].append(article_data)
                else:
                    title_category_dict[category] = [article_data]
    logging.info(f'Seggregated titles_list category-wise.')
    return title_category_dict