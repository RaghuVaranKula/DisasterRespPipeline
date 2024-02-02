import sys
import pandas as pd
from sqlalchemy import create_engine,inspect

def load_data(messages_filepath, categories_filepath):
    """
    Load and merge messages and categories datasets from specified file paths.

    Parameters:
    - messages_filepath (str): File path for the messages CSV file.
    - categories_filepath (str): File path for the categories CSV file.

    Returns:
    - df (DataFrame): A pandas DataFrame containing merged content of messages and categories datasets,
                      merged on the common 'id' column.

    The function reads the messages and categories datasets from their respective file paths,
    then merges them into a single DataFrame on the 'id' column which is common to both datasets.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, left_on='id', right_on='id', how='outer')
    return df

def clean_data(df):
    """
    Cleans the merged DataFrame by splitting categories into separate columns, 
    converting data types, and removing duplicates.

    Parameters:
    - df (DataFrame): The merged DataFrame containing messages and categories data.

    Returns:
    - df (DataFrame): The cleaned DataFrame with categories split into separate columns,
                      data types standardized, and duplicates removed.

    This function assumes that the 'categories' column in the DataFrame contains 
    multiple categories separated by semicolons, which it splits into separate 
    boolean columns. It also ensures there are no duplicate entries based on message ids.
    """
    #Split the values in the categories column on the ';' 
    #character so that each value becomes a separate column.
    categories = df.categories.str.split(';', expand = True)
    row = categories.loc[0] # use this row to extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x[:-2]).values.tolist()
    categories.columns = category_colnames
    categories.related.loc[categories.related == 'related-2'] = 'related-1' # replace all 2 with 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = pd.to_numeric(categories[column])
    # drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace = True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    # check number of duplicates
    if(len(df) - df.id.nunique() != 0):
        df.drop_duplicates(subset = 'id', inplace = True)
    return df

def save_data(df, database_filename):
    """
    Saves the cleaned DataFrame to a SQLite database.

    Parameters:
    - df (DataFrame): The cleaned DataFrame to be saved.
    - database_filename (str): The file path of the SQLite database where the DataFrame will be saved.

    The function creates a SQLite database (if it doesn't exist) and saves the DataFrame
    into a table named 'CleanedData'. It overwrites any existing table with the same name.
    """
    engine = create_engine(f'sqlite:///' + database_filename)
    df.to_sql('Message', engine, index=False, if_exists='replace')
    # Create an inspector object
    inspector = inspect(engine)

    if(inspector.has_table('Message')):
        print("DB created successfully")
    else:
        print("Failed to create DB")

def main():
    """
    The main function of a data processing script that loads, cleans, and saves disaster response data.

    This function orchestrates the data processing pipeline by performing the following steps:
    1. Checks if the correct number of command-line arguments (paths) are provided.
    2. Loads the disaster messages and categories data from specified file paths.
    3. Cleans the combined data.
    4. Saves the cleaned data to a SQLite database at the specified file path.

    The function requires exactly three command-line arguments to run successfully:
    - The first argument is the file path to the disaster messages data file.
    - The second argument is the file path to the disaster categories data file.
    - The third argument is the file path to the SQLite database where the cleaned data will be stored.

    If the correct number of arguments is not provided, the function prints an error message with an example command.

    Parameters:
    - None

    Returns:
    - None
    """
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