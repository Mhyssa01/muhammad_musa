
# 🎬 VibeFlicks - Smart Movie Recommendation System

VibeFlicks is a content-based movie recommendation system built with **Python**, **Scikit-learn**, and **Tkinter**. It helps users discover top-rated movies similar to their search criteria using Natural Language Processing (NLP) and Cosine Similarity. The application features an intuitive GUI with optional dark mode for a better viewing experience.

---

## 🚀 Features

- 🔍 Search movies based on:
  - Title
  - Genre
  - Overview keywords
  - Tagline
  - Production company
  - Original language
- 🧠 Smart recommendation using cosine similarity on combined metadata
- 🌗 Toggle between light and dark mode
- 📊 Treeview result table showing title, rating, and release date
- 🧹 Automatic text cleaning and preprocessing for better results

---

## 🗃 Dataset

- **Source**: `top_1000_popular_movies_tmdb.csv`
- Assumes the dataset includes columns like:
  - `title`
  - `genres`
  - `overview`
  - `tagline`
  - `production_companies`
  - `original_language`
  - `vote_average`
  - `release_date`

---

## ⚙️ How It Works

| Step | Description |
|------|-------------|
| **1. Preprocessing** | The dataset is cleaned by extracting text from nested structures and converting them into a combined `tags` column. |
| **2. Vectorization** | The `tags` are converted into numeric vectors using `CountVectorizer` with stop-word removal. |
| **3. Similarity Calculation** | The user's input is vectorized and compared to all movie vectors using **cosine similarity**. |
| **4. Threshold Filtering** | Movies below a similarity threshold of 0.15 are excluded to avoid poor matches. |
| **5. Display** | Results are sorted by similarity and shown in a scrollable GUI table. |

---

## 🧠 Technologies Used

| Category           | Library/Tool       |
|--------------------|--------------------|
| Data Handling      | `pandas`           |
| NLP & Similarity   | `scikit-learn`     |
| Parsing            | `ast.literal_eval` |
| GUI Development    | `tkinter`          |

---

## 💻 How to Run

1. Ensure you have Python 3.x installed.
2. Install required libraries:
   ```bash
   pip install pandas scikit-learn
   ```
3. Place the dataset `top_1000_popular_movies_tmdb.csv` in the same directory as the script.
4. Run the script:
   ```bash
   python movie_recommender.py
   ```

---

## 🎨 Screenshots

| Light Mode | Dark Mode |
|------------|-----------|
| *(Insert GUI screenshot here)* | *(Insert dark mode screenshot here)* |

---

## 📌 Notes

- Cosine similarity threshold is set to **0.15** to balance between relevance and variety.
- Search results are limited to the top 1000 most similar matches for performance.
- All inputs are optional, but at least one must be filled to trigger a search.

---

## 🛠 Future Improvements

- Add movie poster display using TMDB API
- Add sorting and filtering by year or rating
- Save search history or favorites
- Improve vectorization with TF-IDF or word embeddings

---

## 📜 License

This project is provided for educational purposes. Feel free to fork, improve, and share!
