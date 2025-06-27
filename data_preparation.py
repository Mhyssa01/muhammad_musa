import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from utils import extract_names

def prepare_dataset(csv_path="top_1000_popular_movies_tmdb.csv"):
    """
    Prepares a movie dataset for content-based recommendation by generating a combined text feature
    and vectorizing it using CountVectorizer.

    Args:
        csv_path (str): Path to the CSV file containing movie data. 
                        Defaults to 'top_1000_popular_movies_tmdb.csv'.

    Returns:
        tuple:
            df (pd.DataFrame): The processed DataFrame with additional cleaned and combined text features.
            cv (CountVectorizer): The fitted CountVectorizer instance.
            vectors_sparse (scipy.sparse matrix): The sparse matrix of count vectors for movie tags.

    The function performs the following steps:
    - Reads a CSV file containing movie metadata.
    - Cleans and normalizes textual fields: title, genres, production companies, overview, tagline, and language.
    - Extracts structured names from JSON-like strings using `extract_names`.
    - Concatenates all cleaned fields into a single 'tags' column.
    - Vectorizes the 'tags' column using CountVectorizer with a vocabulary limit of 8000 and English stop words.
    """
    df = pd.read_csv(csv_path)

    df['genres_clean'] = df['genres'].apply(extract_names)
    df['companies_clean'] = df['production_companies'].apply(extract_names)
    df['overview_clean'] = df['overview'].fillna('').str.lower()
    df['tagline_clean'] = df['tagline'].fillna('').str.lower()
    df['language_clean'] = df['original_language'].fillna('').str.lower()
    df['title_clean'] = df['title'].fillna('').str.lower()

    df['tags'] = (
        df['title_clean'] + ' ' +
        df['genres_clean'] + ' ' +
        df['companies_clean'] + ' ' +
        df['overview_clean'] + ' ' +
        df['tagline_clean'] + ' ' +
        df['language_clean']
    )

    cv = CountVectorizer(max_features=8000, stop_words='english')
    vectors_sparse = cv.fit_transform(df['tags'])

    return df, cv, vectors_sparse
