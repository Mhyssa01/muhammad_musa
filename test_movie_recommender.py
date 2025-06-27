"""
Unit tests for the VibeFlicks movie recommendation system.

This module verifies the correctness of:
- Name extraction from JSON-like strings.
- Dataset preparation and structure.
- Movie search logic including matching and no-match behavior.

It tests functions from:
- `utils.py` (extract_names)
- `data_preparation.py` (prepare_dataset)
- `movie_search.py` (search_movies)

Run this file to execute all tests.
"""

import unittest 
import pandas as pd
from utils import extract_names
from data_preparation import prepare_dataset
from movie_search import search_movies

class TestMovieRecommender(unittest.TestCase):
    """
    Test suite for the core functionality of the VibeFlicks recommendation system.

    Includes tests for:
    - Robustness of `extract_names()` against valid and invalid inputs.
    - Structural integrity of the dataset prepared by `prepare_dataset()`.
    - Correctness and reliability of `search_movies()` for exact and unmatched queries.
    """

    def test_extract_names_valid(self):
        """Test that extract_names correctly parses a valid JSON-like genre/company string."""
        input_str = "[{'id': 1, 'name': 'Action'}, {'id': 2, 'name': 'Adventure'}]"
        expected = "Action Adventure".replace("", "")
        result = extract_names(input_str)
        self.assertEqual(result, expected)

    def test_extract_names_invalid(self):
        """Test that extract_names handles malformed or invalid string inputs gracefully."""
        input_str = "not a list"
        result = extract_names(input_str)
        self.assertEqual(result, "")  # Should safely return empty string

    def test_prepare_dataset_structure(self):
        """Test that the dataset is correctly processed and vectors are aligned with rows."""
        df, cv, vectors = prepare_dataset()
        self.assertIn('tags', df.columns)
        self.assertEqual(vectors.shape[0], df.shape[0])
        self.assertGreater(vectors.shape[1], 0)

    def test_search_movies_exact_match(self):
        """Test that search_movies returns non-empty results for a known movie title."""
        df, cv, vectors = prepare_dataset()
        sample_title = df.iloc[0]['title_clean']
        results = search_movies(df, cv, vectors, sample_title, "", "", "", "", "")
        self.assertIsInstance(results, pd.DataFrame)
        self.assertFalse(results.empty)

    def test_search_movies_no_match(self):
        """Test that search_movies returns an empty DataFrame for a non-existent query."""
        df, cv, vectors = prepare_dataset()
        results = search_movies(df, cv, vectors, "nonexistentmovieqwerty", "", "", "", "", "")
        self.assertTrue(results.empty)

if __name__ == "__main__":
    unittest.main()
