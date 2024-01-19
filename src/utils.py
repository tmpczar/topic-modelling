import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import pandas as pd
import numpy as np
from typing import List

# Lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def preprocess_data(text: str) -> str:
    """
    Preprocesses a given text string.

    Args:
        text (str): The input text to be preprocessed.

    Returns:
        str: The preprocessed text after lowercasing, removing special characters,
             tokenization, stop word removal, and lemmatization.
    """
    # Lowercased
    text = text.lower()

    # Removing special character
    text = re.sub(r"\W", " ", text)

    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Iterate through tokens, removing stop words and lemmatizing them
    processed_text = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
    ]

    return " ".join(processed_text)


def prepare_data(filepath: str, sep=";") -> List[str]:
    """
    Reads and prepares data for preprocessing.
    This function is "hardcoded" specifically of this assignment,
    i.e. column names are known, expected and that we wish to concatenate them

    Args:
        filepath (str): Location of the data, a .csv file
        sep (str, optional): _description_. Defaults to ';'.

    Returns:
        pd.DataFrame: Prepared data
    """
    DF = pd.read_csv(filepath, sep=sep).drop(labels="Unnamed: 0", axis=1)
    # NaN replaced with nothing for concatenation
    DF = DF.replace({np.nan: ""})
    DF["qa"] = (
        DF["question_title"] + " " + DF["question_content"] + " " + DF["best_answer"]
    )

    docs = DF["qa"].tolist()
    return docs


def format_topic_string(input_string: str) -> str:
    """Formats topics found by BERTopic in a more readable way"""
    # Remove numbers from the string
    string_without_numbers = re.sub(r"\d+", "", input_string)
    # Replace underscores with spaces
    formatted_string = string_without_numbers.replace("_", " ")
    formatted_string = formatted_string.strip()
    return formatted_string
