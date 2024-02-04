# import libraries
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
import subprocess
import re
import nltk 
nltk.download(['punkt','wordnet','averaged_perceptron_tagger'])

import pandas as pd
import pickle # For serializing and deserializing Python object structures.
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC # SVM model for classification tasks.
from sklearn.multiclass import OneVsRestClassifier # Strategy for multi-class classification.

#"""
#Install Dask if not installed previously
def install_package(package_name):
    """
    Installs a Python package using pip within a Python script.

    Parameters:
    - package_name (str): The name of the package to install.

    Returns:
    - None
    """
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

install_package("dask")
import dask
#"""

def load_data(database_filepath):
    """
    Loads data from a SQLite database and returns feature and target datasets.

    Parameters:
    - database_filepath (str): The filesystem path to the SQLite database file.

    Returns:
    - X (pandas.DataFrame): The features dataset extracted from the database.
    - y (pandas.DataFrame): The target variable dataset extracted from the database.

    """

    # load data from database
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table("Message", engine)
    X = df['message']
    y = df.iloc[:,4:]
    y['related'].replace(2, 1, inplace=True)
    category_names = list(df.columns[4:])
    return X, y, category_names
    

def tokenize(text):
    """
    Tokenizes text into words.

    Parameters:
    - text (str): The text to be tokenized.

    Returns:
    - list[str]: A list of words tokenized and lemmatized from the text.
    """
     
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) #Normalize text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

# Define a Dask delayed function to wrap the GridSearchCV
@dask.delayed
def build_evaluate_save_model(X_train, y_train,X_test,y_test, category_names,model_filepath):
    """
    Builds, evaluates, and saves a machine learning model using GridSearchCV for hyperparameter tuning.

    This function constructs a machine learning pipeline consisting of a text vectorizer, TF-IDF transformer,
    and a multi-output classifier wrapped in GridSearchCV to find the best model parameters based on cross-validation.
    After training and finding the best estimator, the function evaluates the model's performance on a test dataset.
    Finally, it serializes the best estimator (pipeline) using pickle and saves it to a specified file path.

    Parameters:
    - X_train (pd.DataFrame): Training feature dataset, expected to be text data for vectorization.
    - y_train (pd.DataFrame): Training target dataset, compatible with multi-output classification.
    - X_test (pd.DataFrame): Test feature dataset, used for model evaluation.
    - y_test (pd.DataFrame): Test target dataset, used for model evaluation.
    - model_filepath (str): File path where the trained model (best estimator) should be saved as a pickle file.

    Returns:
    - None: This function does not return a value but prints out the best hyperparameters and saves the best model to disk.

    Prints:
    - classification_report and accuracy_score of the model
    - Best hyperparameters found by GridSearchCV.
    - Confirmation message once the model is saved to the specified file path.

    Raises:
    - ValueError: If an error occurs during the pipeline construction, training, or saving process.

    Note:
    This function uses Dask's @delayed decorator for potential parallel computation and efficiency improvements. Ensure
    Dask is properly set up if planning to utilize its parallel computing capabilities.
    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC(dual=False, random_state = 0))))
    ])
    parameters = {
                'tfidf__smooth_idf':[True, False],
                'clf__estimator__estimator__C': [1, 2, 5]
             }
    cv = GridSearchCV(pipeline, param_grid=parameters, cv = 2,n_jobs=-1,verbose=2)

    print('Training model...')
    cv.fit(X_train, y_train)
    
    print('Evaluating model...')
    y_pred = cv.predict(X_test)
    print("Accuracy")
    print((y_pred == y_test).mean())

    for i in range(len(y_test.columns)):
        print(classification_report(y_test.iloc[:,i], y_pred[:,i]))
        print(category_names[i],accuracy_score(y_test.iloc[:,i], y_pred[:,i]))
        print('---------------------------------')

    print("\nBest Parameters:", cv.best_params_)

    print('\nSaving model...\n    MODEL: {}'.format(model_filepath))
    pickle.dump(cv, open(model_filepath, 'wb')) #dumping pickle file
    
    return cv

def main():
    """
    Main function to orchestrate the data loading, model training, evaluation, and saving.
    It specifically handles the following tasks:
    - Loading the data from a predefined source.
    - Building a machine learning model with predefined parameters.
    - Training the model on the loaded data.
    - Evaluating the model's performance on a test dataset.
    - Saving the trained model to disk for future use.

    Raises:
    - Exception: Describes any exceptions that might be raised during the execution
      of the function, such as issues with data loading, model training failures,
      or errors during model saving.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
        
        print("Building Model...")
        # Call the delayed function of Dask
        result = build_evaluate_save_model(X_train, Y_train,X_test,Y_test, category_names,model_filepath)

        # Compute the result with Dask using the threaded scheduler
        result.compute(scheduler='threads')

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()