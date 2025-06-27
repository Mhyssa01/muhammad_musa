import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity

def search_movies(df, cv, vectors_sparse, title, genres, language, overview, tagline, company):
    """
    Searches for similar movies based on user input using content-based filtering.

    Args:
        df (pd.DataFrame): The preprocessed movie dataset with cleaned textual fields and tag combinations.
        cv (CountVectorizer): Fitted CountVectorizer used to transform text into feature vectors.
        vectors_sparse (scipy.sparse matrix): Sparse matrix of tag vectors for all movies in the dataset.
        title (str): Title or partial title of the movie.
        genres (str): Genre(s) specified by the user.
        language (str): Original language of the movie (ISO code or name).
        overview (str): Keywords or phrases from the movie's description/overview.
        tagline (str): Keywords from the movie's tagline.
        company (str): Name of the production company.

    Returns:
        pd.DataFrame: A DataFrame containing the top recommended movies based on cosine similarity.
                      Includes 'title', 'vote_average', and 'release_date' columns.
                      Returns an empty DataFrame if no matches exceed the similarity threshold.

    Logic:
        - Filters movies by language if provided.
        - If only language is specified, returns top 1000 movies in that language.
        - Combines all non-empty query fields into a single search string.
        - Transforms the query into a vector and computes cosine similarity with all movie vectors.
        - Returns up to 1000 movies that exceed a minimum similarity threshold (default: 0.15),
          sorted by similarity score in descending order.
    """
    query = f"{title} {genres} {overview} {tagline} {company}".lower()
    filtered_df = df[df['language_clean'] == language.strip().lower()] if language else df

    if not any([title, genres, overview, tagline, company]) and language:
        return filtered_df[['title', 'vote_average', 'release_date']].head(1000)

    query_vector = cv.transform([query])
    cosine_sim = cosine_similarity(query_vector, vectors_sparse).flatten()

    threshold = 0.15
    filtered_indices = filtered_df.index.tolist()
    valid_indices = [i for i in filtered_indices if cosine_sim[i] >= threshold]

    if not valid_indices:
        return pd.DataFrame()

    sorted_indices = sorted(valid_indices, key=lambda x: cosine_sim[x], reverse=True)[:1000]
    return df.iloc[sorted_indices][['title', 'vote_average', 'release_date']]
