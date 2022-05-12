import sys
import pandas as pd
from sqlalchemy import create_engine

# Sequence of functions that make the ETL step

# 1. Extract
def load_data(messages_filepath, categories_filepath):
    '''Function that loads the data

    :param messages_filepath: file path of messages dataset
    :param categories_filepath: file path of categories dataset
    :return: merged dataframe of messages and categories datasets
    '''

    # load messages and categories datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, how='left', on='id')
    return df

# 2. Transform
def clean_data(df):
    '''Function that clears the dataframe that came from the previous function - load_data()

    :param df: dataframe that came from the previous function
    :return: final cleaned dataframe
    '''

    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";", expand=True)
    row = categories.iloc[0]
    category_colnames = [i[:-2] for i in row]
    categories.columns = category_colnames

    # convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)

    # replace categories column in df with new category columns
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.merge(df, categories, left_index=True, right_index=True)

    # remove duplicates
    df.drop_duplicates(inplace=True)
    return df

# 3. Load
def save_data(df, database_filename):
    ''' Function that save the final cleaned dataframe that came from the previous function - clean_data()
    and save in a SQL database

    :param df: final cleaned dataframe that came from the previous function
    :param database_filename: the name of your file that you need to upload in the database
    :return: none
    '''

    # upload the final dataframe into a SQL database
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql(database_filename, engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()