import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Sequence of functions that make the machine learning pipeline

def load_data(database_filepath):
    '''Function that uploads the table that is in the database created in the previous ETL step.

    :param database_filepath: the file path of your database
    :return: our text variables (X) and our response variables (Y)
    '''
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table("labeledmessages", engine)

    # Small pre-processes before starting
    # 1. passes label values '2' to '0' in target variable 'related'
    df.loc[df['related'] == 2, 'related'] = 0

    # 2. drop the 'child_alone' variable for lack of representation
    df.drop(['child_alone'], axis=1, inplace=True)

    # 3. divide train_set in X/y
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    return X, Y

def tokenize(text):
    '''Function that cleans, tokenizes, and lemmatizes our text dataset so that it is
    ready to feed into machine learning models.

    :param text: text dataset (X) that came from the previous function - load_data()
    :return: set of texts prepared to feed the machine learning algorithm
    '''
    # clean text
    text = re.sub('\[.*?\]', '', str(text))
    text = re.sub('https?://\S+|www\.\S+', '', str(text))  # Remove as urls
    text = re.sub('<.*?>+', '', str(text))
    text = re.sub('[%s]' % re.escape(string.punctuation), '', str(text))  # Remove as pontuações
    text = re.sub('\n', '', str(text))
    text = re.sub('\w*\d\w*', '', str(text))
    text = re.sub("\W", " ", str(text).lower().strip())

    # instantiate the tokens and stopwords
    tokens = word_tokenize(str(text))
    stop_words = stopwords.words("english")

    # instantiate the lemmatizer
    lemmatizer = WordNetLemmatizer()

    # apply the final list with clean tokens
    clean_tokens = []
    for tok in tokens:
        if tok not in stop_words:
            clean_tok = lemmatizer.lemmatize(tok)
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()