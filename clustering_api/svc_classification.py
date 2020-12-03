import requests
import logging
import json
import pandas as pd
import pickle
import numpy as np

from tqdm import tqdm 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from utils import get_summary

logging.basicConfig( format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='[%Y-%m-%d %H:%M:%S +0000]')

punctuation_signs = list("?:!.,;")
stop_words = list(stopwords.words('english'))

# Codes will change as when the model is updated.
category_codes = {
    0: 'business',
    1: 'entertainment',
    2: 'politics',
    3: 'sport',
    4: 'tech',
    5: 'other'
    }

#Loading Models
model_path = "Pickles/"

# SVM
path_svm = model_path + 'best_svc.pickle'
with open(path_svm, 'rb') as data:
    svc_model = pickle.load(data)

path_tfidf = model_path + 'tfidf.pickle'
with open(path_tfidf, 'rb') as data:
    tfidf = pickle.load(data)

def get_feature_df(titles_list):
    news_contents = []
    list_titles = []
    list_links = []
    list_sources = []
    for item in tqdm(titles_list, desc= 'Scraping content from URLs', ncols=100):
        list_titles.append(item['Title'])
        list_links.append(item['Href'])
        list_sources.append(item['Source'])
        news_contents.append(get_summary(item['Href']))

    df_features = pd.DataFrame(
            {'Content': news_contents 
            })

    # df_show_info
    df_show_info = pd.DataFrame(
        {'Article Title': list_titles,
        'Article Link': list_links,
        'Content': news_contents,
        'Source': list_sources})   

    return df_features, df_show_info     

def create_features_from_df(df):
    
    df['Content_Parsed_1'] = df['Content'].str.replace("\r", " ")
    df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace("\n", " ")
    df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace("    ", " ")
    df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace('"', '')
    
    df['Content_Parsed_2'] = df['Content_Parsed_1'].str.lower()
    
    df['Content_Parsed_3'] = df['Content_Parsed_2']
    for punct_sign in punctuation_signs:
        df['Content_Parsed_3'] = df['Content_Parsed_3'].str.replace(punct_sign, '')
        
    df['Content_Parsed_4'] = df['Content_Parsed_3'].str.replace("'s", "")
    
    wordnet_lemmatizer = WordNetLemmatizer()
    nrows = len(df)
    lemmatized_text_list = []
    for row in range(0, nrows):

        # Create an empty list containing lemmatized words
        lemmatized_list = []
        # Save the text and its words into an object
        text = df.loc[row]['Content_Parsed_4']
        text_words = text.split(" ")
        # Iterate through every word to lemmatize
        for word in text_words:
            lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
        # Join the list
        lemmatized_text = " ".join(lemmatized_list)
        # Append to the list containing the texts
        lemmatized_text_list.append(lemmatized_text)
    
    df['Content_Parsed_5'] = lemmatized_text_list
    
    df['Content_Parsed_6'] = df['Content_Parsed_5']
    for stop_word in stop_words:
        regex_stopword = r"\b" + stop_word + r"\b"
        df['Content_Parsed_6'] = df['Content_Parsed_6'].str.replace(regex_stopword, '')
        
    df = df['Content_Parsed_6']
    df = df.rename(columns={'Content_Parsed_6': 'Content_Parsed'})
    
    # TF-IDF
    features = tfidf.transform(df).toarray()
    
    return features

def predict_from_features(features):    
    # Obtain the highest probability of the predictions for each article
    predictions_proba = svc_model._predict_proba(features)    
    predictions_proba_max = predictions_proba.max(axis=1)   
    # Predict using the input model
    # Replace prediction with 6 if associated cond. probability less than threshold
    predictions = []

    for prob_arr in predictions_proba:
        # max_prob = np.amax(prob_arr, axis=0)
        # if max_prob > .65:
        #     cat = np.argmax(prob_arr, axis=0)
        #     predictions.append(cat)
        # else:
        #     predictions.append(5)
        cat = np.argmax(prob_arr, axis=0)
        predictions.append(cat)

    # Return result
    categories = [category_codes[x] for x in predictions]
    
    return categories, predictions_proba_max