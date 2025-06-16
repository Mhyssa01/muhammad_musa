import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def search_movies(df, cv, vectors_sparse, title, genres, language, overview, tagline, company):
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
