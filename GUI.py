# ===== Import Required Libraries =====
import pandas as pd
import ast  # For safely evaluating strings containing Python expressions
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import messagebox, ttk

# ===== Load and Prepare Dataset =====
df = pd.read_csv("top_1000_popular_movies_tmdb.csv")

# Function to extract names (e.g., genres, companies) from stringified lists
def extract_names(x):
    try:
        return " ".join([i['name'].replace(" ", "") for i in ast.literal_eval(x)])
    except:
        return ""

# Apply cleaning to multiple text fields
df['genres_clean'] = df['genres'].apply(extract_names)
df['companies_clean'] = df['production_companies'].apply(extract_names)
df['overview_clean'] = df['overview'].fillna('').str.lower()
df['tagline_clean'] = df['tagline'].fillna('').str.lower()
df['language_clean'] = df['original_language'].fillna('').str.lower()
df['title_clean'] = df['title'].fillna('').str.lower()

# Combine relevant text fields into one 'tags' column for vectorization
df['tags'] = (
        df['title_clean'] + ' ' +
        df['genres_clean'] + ' ' +
        df['companies_clean'] + ' ' +
        df['overview_clean'] + ' ' +
        df['tagline_clean'] + ' ' +
        df['language_clean']
)

# ===== Vectorize Tags Column =====
cv = CountVectorizer(max_features=8000, stop_words='english')
vectors_sparse = cv.fit_transform(df['tags'])  # Sparse matrix for performance

# ===== Search Logic with Cosine Similarity and Language Filter =====
def search_movies(title, genres, language, overview, tagline, company):
    # Combine inputs into a query string
    query = f"{title} {genres} {overview} {tagline} {company}".lower()

    # Filter dataset by selected language if provided
    filtered_df = df[df['language_clean'] == language.strip().lower()] if language else df

    # Return top movies in the selected language if no query input is provided
    if not any([title, genres, overview, tagline, company]) and language:
        return filtered_df[['title', 'vote_average', 'release_date']].head(1000)

    # Compute cosine similarity between query and movie dataset
    query_vector = cv.transform([query])
    cosine_sim = cosine_similarity(query_vector, vectors_sparse).flatten()

    # Filter results by similarity threshold and original language condition
    threshold = 0.15
    filtered_indices = filtered_df.index.tolist()
    valid_indices = [i for i in filtered_indices if cosine_sim[i] >= threshold]

    if not valid_indices:
        return pd.DataFrame()  # Return empty DataFrame if no match

    # Sort by similarity and return top 50 results
    sorted_indices = sorted(valid_indices, key=lambda x: cosine_sim[x], reverse=True)[:1000]
    return df.iloc[sorted_indices][['title', 'vote_average', 'release_date']]

# ===== Graphical User Interface (GUI) =====
def run_gui():
    root = tk.Tk()
    root.title("🎬 Movie Recommender System")
    root.geometry("1020x720")
    root.configure(bg="#FFF3E0")  # Light theme background by default

    # Variable to toggle dark mode
    dark_mode = tk.BooleanVar(value=False)

    # Dark/Light mode toggle function
    def toggle_dark():
        if dark_mode.get():  # If dark mode enabled
            root.configure(bg="#212121")
            input_frame.configure(bg="#424242")
            result_frame.configure(bg="#303030")
            for widget in input_frame.winfo_children():
                if isinstance(widget, tk.Label):
                    widget.configure(bg="#424242", fg="#ffffff")
                elif isinstance(widget, tk.Entry):
                    widget.configure(bg="#616161", fg="#ffffff")
            style.configure("Treeview", background="#424242", foreground="#ffffff",
                            fieldbackground="#424242")
        else:  # Switch back to light mode
            root.configure(bg="#FFF3E0")
            input_frame.configure(bg="#FFE0B2")
            result_frame.configure(bg="#FFF3E0")
            for widget in input_frame.winfo_children():
                if isinstance(widget, tk.Label):
                    widget.configure(bg="#FFE0B2", fg="#4E342E")
                elif isinstance(widget, tk.Entry):
                    widget.configure(bg="#FFFFFF", fg="#000000")
            style.configure("Treeview", background="#FFFDE7", foreground="#000000",
                            fieldbackground="#FFFDE7")

    # ===== Style Configuration =====
    font_main = ("Segoe UI", 11)
    font_label = ("Segoe UI", 10, "bold")
    label_fg = "#4E342E"
    button_bg = "#FF9800"
    button_fg = "#ffffff"

    # ===== Input Frame =====
    input_frame = tk.LabelFrame(root, text="Search Criteria", bg="#FFE0B2", font=font_label, fg="#BF360C")
    input_frame.pack(padx=20, pady=20, fill="x")

    # Labels and Input Fields
    labels = ["🌐 Language:", "🎞 Title:", "📚 Genre:",
              "🧾 Overview Keywords:", "🗣 Tagline:", "🏢 Production Company:"]
    entries = []

    for i, label_text in enumerate(labels):
        row = i // 2
        col = (i % 2) * 2
        label = tk.Label(input_frame, text=label_text, font=font_label, fg=label_fg, bg="#FFE0B2")
        label.grid(row=row, column=col, sticky="w", padx=10, pady=5)
        entry = tk.Entry(input_frame, width=35, font=font_main, bg="#FFFFFF", fg="#000000",
                         relief=tk.SOLID, bd=2, highlightthickness=1, highlightcolor="#FFFFFF")
        entry.grid(row=row, column=col + 1, padx=5, pady=5)
        entries.append(entry)

    # ===== Output Frame =====
    result_frame = tk.LabelFrame(root, text="🎯 Recommendations", bg="#FFF3E0", font=font_label, fg="#E65100")
    result_frame.pack(padx=20, pady=10, fill="both", expand=True)

    # Treeview with Scrollbar
    tree_scroll = tk.Scrollbar(result_frame)
    tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    tree = ttk.Treeview(result_frame, columns=("Title", "Rating", "Release Date"), show="headings",
                        yscrollcommand=tree_scroll.set, height=15)
    tree.heading("Title", text="🎬 Title")
    tree.heading("Rating", text="⭐ Rating")
    tree.heading("Release Date", text="🗓️ Release Date")
    tree.column("Title", width=400)
    tree.column("Rating", width=100, anchor="center")
    tree.column("Release Date", width=120, anchor="center")
    tree.pack(fill="both", expand=True)
    tree_scroll.config(command=tree.yview)

    # Style for Treeview
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"), foreground="#BF360C")
    style.configure("Treeview", font=("Segoe UI", 10), background="#FFFDE7", fieldbackground="#FFFDE7")

    # ===== Search Button Function =====
    def on_search():
        tree.delete(*tree.get_children())  # Clear previous results
        inputs = [e.get().strip() for e in entries]
        if all(val == "" for val in inputs):
            messagebox.showwarning("Input Required", "⚠️ Please enter at least one search field.")
            return

        results = search_movies(inputs[1], inputs[2], inputs[0], inputs[3], inputs[4], inputs[5])
        if results.empty:
            messagebox.showinfo("No Results", "🔍 No results found. Try refining your input.")
        else:
            for _, row in results.iterrows():
                tree.insert("", tk.END, values=(row['title'], row['vote_average'], row['release_date']))

    # ===== Search Button =====
    tk.Button(root, text="🔍 Search", command=on_search,
              bg=button_bg, fg=button_fg, font=("Segoe UI", 11, "bold"),
              relief=tk.FLAT, padx=20, pady=8, cursor="hand2", activebackground="#FB8C00").pack(pady=10)

    # ===== Dark Mode Toggle Checkbox =====
    tk.Checkbutton(root, text="🌙 Dark Mode", variable=dark_mode, onvalue=True, offvalue=False,
                   command=toggle_dark, bg="#FFF3E0", font=font_main).pack()

    # ===== Start GUI Loop =====
    root.mainloop()

# ===== Run the Movie Recommender Application =====
run_gui()
