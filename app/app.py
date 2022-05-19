# Import necessary packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import string
import pickle
import streamlit as st
from sqlalchemy import create_engine

# Load data
engine = create_engine('sqlite:///../data/disastersresponse.db')
df = pd.read_sql_table("labeledmessages", engine)

####### Streamlit Configuration #######
# 1. Page initial configuration
st.set_page_config(page_title='Disaster Response', page_icon=None, layout="centered",
                   initial_sidebar_state="auto", menu_items=None)


# 2. Import saved model from 'train_classifier.py'
# model = pickle.load(open('new_model.pkl','rb'))

def main():
    # 3. Web app title configuration and the firsts paragraphs to explain the app
    st.title("Disaster Response Project")

    """
    Analyzing message data for disaster response
    """

    # 4. Configuration of text input
    phrase = st.text_input('Enter a message to classify')

    # 5. Our inputs
    # inputs = [[phrase]]

    # 6. Making and printing our prediction
    st.button('See message categories')
    #       result = model.predict(inputs)
    #       updated_res = result.flatten().astype(float)
    #       st.success('Your Movie Revenue is: {}'.format(updated_res))


# 7. Calling main method
if __name__ == '__main__':
    main()

# 8. Making informative graphs
st.title("Data Exploration")
