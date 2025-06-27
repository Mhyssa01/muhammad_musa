"""
Main script to launch the VibeFlicks movie recommendation application.

This script performs the following:
1. Loads and prepares the movie dataset using the `prepare_dataset` function from `data_preparation`.
2. Initializes and runs the Tkinter-based GUI using the `run_gui` function from `gui`.

The recommendation system is content-based and allows users to search for movies by various metadata fields
(e.g., title, genre, language, tagline, etc.), returning similar movies based on textual similarity.
"""

from data_preparation import prepare_dataset 
from gui import run_gui

if __name__ == "__main__":
    # Load and process the dataset, then launch the GUI.
    df, cv, vectors_sparse = prepare_dataset()
    run_gui(df, cv, vectors_sparse)
