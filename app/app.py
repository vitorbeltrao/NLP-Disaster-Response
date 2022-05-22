# Import necessary packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
import re
import string

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


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


# Load data
df = pd.read_csv('https://raw.githubusercontent.com/vitorbeltrao/NLP-Disaster-Response/master/app/labeledmessages.csv', low_memory=False)

# Small pre-processes before starting
# 1. passes label values '2' to '0' in target variable 'related'
df.loc[df['related'] == 2, 'related'] = 0

# 2. drop the 'child_alone' variable for lack of representation
df.drop(['child_alone'], axis=1, inplace=True)

####### Streamlit Configuration #######
# 1. Page initial configuration
st.set_page_config(page_title='Disaster Response', page_icon=None, layout="centered",
                   initial_sidebar_state="auto", menu_items=None)

# 2. Import saved model from 'train_classifier.py'
model = joblib.load(open('lgbm_model.pkl','rb'))


def main():
    # 3. Web app title configuration and the firsts paragraphs to explain the app
    st.title("Disaster Response Project")

    """
    Analyzing message data for disaster response
    """

    # 4. Configuration of text input
    text_message = st.text_input("Your message")

    # 5. Our inputs
    inputs = [text_message]

    # 6. Making and printing our prediction
    predict_button = st.button('See message categories')

    if predict_button:
        # The final model prediction
        result = model.predict(inputs)[0]
        updated_res = dict(zip(df.columns[4:], result))

        # dict iterate that shows only true categories
        for k, v in updated_res.items():
            if v == 1:
                st.success('{}'.format(k))


# 7. Calling main method
if __name__ == '__main__':
    main()

# 8. Making informative graphs
st.title("Data Exploration")

df_filtered = df.drop(columns=['id', 'message', 'original', 'genre'])
categories = (df_filtered.columns)
df_filtered['sum'] = df_filtered.sum(axis=1)

# Sidebar Configuration
st.sidebar.header("Select one of the labels to see its respective frequency graph:")

# Select box
labels = st.sidebar.selectbox(
    'Select Label:',
    ('related', 'request', 'offer', 'aid_related', 'medical_help',
     'medical_products', 'search_and_rescue', 'security', 'military',
     'water', 'food', 'shelter', 'clothing', 'money', 'missing_people',
     'refugees', 'death', 'other_aid', 'infrastructure_related',
     'transport', 'buildings', 'electricity', 'tools', 'hospitals',
     'shops', 'aid_centers', 'other_infrastructure', 'weather_related',
     'floods', 'storm', 'fire', 'earthquake', 'cold', 'other_weather',
     'direct_report'))

# Show dataframe
agree = st.sidebar.checkbox('Select the checkbox if you want to see the entire dataset')
if agree:
    st.dataframe(data=df, width=None, height=None)

# Plot the graphs
fig, ax = plt.subplots()
sns.countplot(labels, data=df_filtered, ax=ax)
ax.set_title(labels + ' Variable Distribution')
st.pyplot(fig)

# Text to help understand the chart
st.text('The x axis has categories "1" and "0"')
st.text('Category 1 represents whether an instance is linked to that label')
st.text('Category 0 represents whether an instance is not linked to that label')


