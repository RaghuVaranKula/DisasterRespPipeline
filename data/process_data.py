import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    input:
        messages_filepath: Load messages.csv into a dataframe .
        categories_filepath: Load categories.csv into a dataframe.
    output:
        df: Merged messages and categories datasets using the common id
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, left_on='id', right_on='id', how='outer')
    return df

def clean_data(df):
    '''
    input:
        df: Merged raw dataset without cleaning.
    output:
        df: Dataset after cleaning.
    '''
    #Split the values in the categories column on the ';' 
    #character so that each value becomes a separate column.
    categories = df.categories.str.split(';', expand = True)
    row = categories.loc[0] # use this row to extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x[:-2]).values.tolist()
    categories.columns = category_colnames
    # categories.related.loc[categories.related == 'related-2'] = 'related-1'
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    # drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace = True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    # check number of duplicates
    if(len(df) - df.id.nunique() != 0):
        df.drop_duplicates(subset = 'id', inplace = True)
    return df

def save_data(df, database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df.to_sql('Message', engine, index=False)
    if(engine.has_table('Message')):
        print("DB created successfully")
    else:
        print("Failed to create DB")

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


    else:
        print('Error: incorrect path or incorrect sequence of paths'\
              '\nExample: python process_data.py disaster_messages.csv '\
              ' disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()