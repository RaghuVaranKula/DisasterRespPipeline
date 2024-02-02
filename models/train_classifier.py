# import libraries
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
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


def build_model():
    """
    Constructs and returns a machine learning model pipeline.

    Returns:
    - model (Pipeline): A scikit-learn Pipeline object that encapsulates the
      preprocessing steps and the classifier with GridSearch.

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
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring='precision_samples', cv = 5,n_jobs=-1,verbose=2)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates and prints the performance of the machine learning model on a test dataset.

    Parameters:
    - model (Pipeline): The machine learning model to evaluate.
    - X_test (pandas.DataFrame): The features of the test dataset.
    - y_test (pandas.DataFrame or pandas.Series): The true labels of the test dataset.
    """

    Y_pred = model.predict(X_test)
    print("Accuracy")
    print((Y_pred == Y_test).mean())
    for i, col in enumerate(category_names):
        print('{} category metrics: '.format(col))
        print(classification_report(Y_test.iloc[:,i], Y_pred[:,i]))
        print(accuracy_score(Y_test.iloc[:,i], Y_pred[:,i]))
    print('---------------------------------')


def save_model(model, model_filepath):
    """
    Serializes and saves the machine learning model to a specified file path.

    Parameters:
    - model (Pipeline): The machine learning model to save.
    - model_filepath (str): The path where the model should be saved, including the filename.
    """

    pickle.dump(model, open(model_filepath, 'wb')) #dumping pickle file


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