import unittest
import pandas as pd
from utils import extract_names
from data_preparation import prepare_dataset
from movie_search import search_movies

class TestMovieRecommender(unittest.TestCase):

    def test_extract_names_valid(self):
        input_str = "[{'id': 1, 'name': 'Action'}, {'id': 2, 'name': 'Adventure'}]"
        expected = "Action Adventure".replace("", "")
        result = extract_names(input_str)
        self.assertEqual(result, expected)

    def test_extract_names_invalid(self):
        input_str = "not a list"
        result = extract_names(input_str)
        self.assertEqual(result, "")  # Should safely return empty string

    def test_prepare_dataset_structure(self):
        df, cv, vectors = prepare_dataset()
        self.assertIn('tags', df.columns)
        self.assertEqual(vectors.shape[0], df.shape[0])
        self.assertGreater(vectors.shape[1], 0)

    def test_search_movies_exact_match(self):
        df, cv, vectors = prepare_dataset()
        # Pick the title of a known movie to simulate user input
        sample_title = df.iloc[0]['title_clean']
        results = search_movies(df, cv, vectors, sample_title, "", "", "", "", "")
        self.assertIsInstance(results, pd.DataFrame)
        self.assertFalse(results.empty)

    def test_search_movies_no_match(self):
        df, cv, vectors = prepare_dataset()
        results = search_movies(df, cv, vectors, "nonexistentmovieqwerty", "", "", "", "", "")
        self.assertTrue(results.empty)

if __name__ == "__main__":
    unittest.main()