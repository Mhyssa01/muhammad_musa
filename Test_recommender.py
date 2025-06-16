import unittest
import pandas as pd
from movie_recommender import extract_names, search_movies

class TestMovieRecommender(unittest.TestCase):

    def test_extract_names_valid(self):
        test_input = "[{'name': 'Warner Bros'}, {'name': 'Pixar'}]"
        expected_output = "WarnerBros Pixar"
        self.assertEqual(extract_names(test_input), expected_output)

    def test_extract_names_invalid(self):
        test_input = "not a list"
        self.assertEqual(extract_names(test_input), "")

    def test_search_no_input(self):
        result = search_movies("", "", "", "", "", "")
        self.assertTrue(result.shape[0] > 0)  # Should return most popular movies

    def test_search_language_filter(self):
        result = search_movies("", "", "en", "", "", "")
        self.assertTrue((result['title'].notnull()).all())

if __name__ == '__main__':
    unittest.main()
