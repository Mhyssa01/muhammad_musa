import tkinter as tk 
from tkinter import messagebox, ttk
import pandas as pd
from movie_search import search_movies

def run_gui(df, cv, vectors_sparse):
    """
    Launches a Tkinter-based graphical user interface (GUI) for a content-based movie recommendation system.

    Args:
        df (pd.DataFrame): The movie dataset containing cleaned and preprocessed fields used for matching.
        cv (CountVectorizer): The CountVectorizer instance used for feature extraction from the movie tags.
        vectors_sparse (scipy.sparse matrix): Sparse matrix representing vectorized movie tags for similarity comparison.

    Features:
        - Entry fields for user to input search parameters: language, title, genre, overview, tagline, and production company.
        - A "Search" button that calls the `search_movies` function to retrieve recommendations.
        - Results displayed in a Treeview with columns for title, rating, and release date.
        - Toggle for Dark Mode to switch UI themes for accessibility and comfort.

    Dependencies:
        - `search_movies`: Function used to filter and rank movies based on similarity.
        - `tkinter`, `ttk`, and `messagebox`: For building and managing the GUI interface.

    Returns:
        None
    """
    root = tk.Tk()
    root.title("🎬 VibeFlicks")
    root.geometry("1020x720")
    root.configure(bg="#FFF3E0")

    dark_mode = tk.BooleanVar(value=False)

    def toggle_dark():
        """Toggle between light and dark UI themes."""
        if dark_mode.get():
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
        else:
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

    font_main = ("Segoe UI", 11)
    font_label = ("Segoe UI", 10, "bold")
    label_fg = "#4E342E"
    button_bg = "#FF9800"
    button_fg = "#ffffff"

    input_frame = tk.LabelFrame(root, text="Search Criteria", bg="#FFE0B2", font=font_label, fg="#BF360C")
    input_frame.pack(padx=20, pady=20, fill="x")

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

    result_frame = tk.LabelFrame(root, text="🎯 Recommendations", bg="#FFF3E0", font=font_label, fg="#E65100")
    result_frame.pack(padx=20, pady=10, fill="both", expand=True)

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

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"), foreground="#BF360C")
    style.configure("Treeview", font=("Segoe UI", 10), background="#FFFDE7", fieldbackground="#FFFDE7")

    def on_search():
        """Triggered when user clicks the Search button. Retrieves and displays recommended movies."""
        tree.delete(*tree.get_children())
        inputs = [e.get().strip() for e in entries]
        if all(val == "" for val in inputs):
            messagebox.showwarning("Input Required", "⚠️ Please enter at least one search field.")
            return

        results = search_movies(df, cv, vectors_sparse, inputs[1], inputs[2], inputs[0], inputs[3], inputs[4], inputs[5])
        if results.empty:
            messagebox.showinfo("No Results", "🔍 No results found. Try refining your input.")
        else:
            for _, row in results.iterrows():
                tree.insert("", tk.END, values=(row['title'], row['vote_average'], row['release_date']))

    tk.Button(root, text="🔍 Search", command=on_search,
              bg=button_bg, fg=button_fg, font=("Segoe UI", 11, "bold"),
              relief=tk.FLAT, padx=20, pady=8, cursor="hand2", activebackground="#FB8C00").pack(pady=10)

    tk.Checkbutton(root, text="🌙 Dark Mode", variable=dark_mode, onvalue=True, offvalue=False,
                   command=toggle_dark, bg="#FFF3E0", font=font_main).pack()

    root.mainloop()
