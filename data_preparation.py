import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from utils import extract_names

def prepare_dataset(csv_path="top_1000_popular_movies_tmdb.csv"):
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
