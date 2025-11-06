# movie_recommender.py
# Requires: networkx, numpy, scikit-learn, matplotlib, tkinter
# Install via: pip install networkx numpy scikit-learn matplotlib

import networkx as nx
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import heapq
import random
import math
import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk, simpledialog


# =====================================================
# 🎥 MOVIE RECOMMENDER CLASS (Graph-Based)
# =====================================================
class MovieRecommender:
    """
    Graph-based hybrid movie recommender system with visualization tools.
    Combines collaborative & content-based filtering using graph traversal.
    """

    def _init_(self):
        self.G = nx.Graph()
        self.users = set()
        self.movies = set()
        self.genres = set()
        self.tags = set()
        self.ratings = []

    # -------------------------- Graph Building --------------------------
    def add_user(self, uid):
        if uid not in self.G:
            self.G.add_node(uid, type='user')
            self.users.add(uid)

    def add_movie(self, mid, meta=None):
        if mid not in self.G:
            self.G.add_node(mid, type='movie', **(meta or {}))
            self.movies.add(mid)

    def add_genre(self, gid):
        if gid not in self.G:
            self.G.add_node(gid, type='genre')
            self.genres.add(gid)

    def add_tag(self, tid):
        if tid not in self.G:
            self.G.add_node(tid, type='tag')
            self.tags.add(tid)

    def add_rating(self, user, movie, rating):
        self.add_user(user)
        self.add_movie(movie)
        self.G.add_edge(user, movie, type='rated', rating=float(rating), weight=float(rating)/5.0)
        self.ratings.append((user, movie, float(rating)))

    def link_movie_genre(self, movie, genre):
        self.add_movie(movie)
        self.add_genre(genre)
        self.G.add_edge(movie, genre, type='genre')

    # -------------------------- Graph Queries --------------------------
    def user_rated_movies(self, user):
        return [n for n in self.G.neighbors(user) if self.G.nodes[n].get('type') == 'movie']

    def movie_users(self, movie):
        return [n for n in self.G.neighbors(movie) if self.G.nodes[n].get('type') == 'user']

    # -------------------------- BFS Candidates --------------------------
    def bfs_candidates(self, user, max_depth=3):
        if user not in self.G:
            return {}
        visited = {user}
        q = deque([(user, 0)])
        candidates = defaultdict(float)

        while q:
            node, depth = q.popleft()
            if depth >= max_depth:
                continue
            for nbr in self.G.neighbors(node):
                if nbr in visited:
                    continue
                visited.add(nbr)
                q.append((nbr, depth + 1))
                if self.G.nodes[nbr].get('type') == 'movie':
                    if not self.G.has_edge(user, nbr):
                        candidates[nbr] += 1.0 / (depth + 1)
        return dict(candidates)

    # -------------------------- Scoring --------------------------
    def score_movie(self, user, movie, weights):
        # Collaborative similarity
        neigh_users = [n for n in self.G.neighbors(movie)
                       if self.G.nodes[n].get('type') == 'user' and n != user]
        sim_score = sum(self.G[u][movie].get('rating', 0) for u in neigh_users)
        count = len(neigh_users)
        sim_norm = (sim_score / (5.0 * count)) if count else 0.0

        # Genre overlap
        user_genres = set()
        for m in self.user_rated_movies(user):
            for g in self.G.neighbors(m):
                if self.G.nodes[g].get('type') == 'genre':
                    user_genres.add(g)
        movie_genres = {g for g in self.G.neighbors(movie)
                        if self.G.nodes[g].get('type') == 'genre'}
        genre_overlap = len(user_genres & movie_genres) / (len(user_genres) + 1e-9) if user_genres else 0.0

        # Popularity
        pop = len(self.movie_users(movie))
        pop_norm = math.sqrt(pop) / math.sqrt(max(1, len(self.users)))

        # User bias (average rating)
        uratings = [self.G[user][m].get('rating')
                    for m in self.user_rated_movies(user)
                    if self.G.has_edge(user, m)]
        user_bias = (sum(uratings) / len(uratings)) / 5.0 if uratings else 0.0

        # Weighted score
        s = (
            weights.get('w_sim', 0.55) * sim_norm +
            weights.get('w_genre', 0.25) * genre_overlap +
            weights.get('w_pop', 0.10) * pop_norm +
            weights.get('w_bias', 0.10) * user_bias
        )
        return s

    # -------------------------- Best-First Recommendation --------------------------
    def recommend_best_first(self, user, K=5, weights=None, max_depth=3):
        if weights is None:
            weights = {'w_sim': 0.55, 'w_genre': 0.25, 'w_pop': 0.10, 'w_bias': 0.10}

        cand = self.bfs_candidates(user, max_depth=max_depth)
        heap = []
        for m in cand.keys():
            heapq.heappush(heap, (-self.score_movie(user, m, weights), m))
        out = []
        while heap and len(out) < K:
            out.append(heapq.heappop(heap)[1])
        return out

    # -------------------------- Visualization --------------------------
    def plot_full_graph(self):
        pos = nx.spring_layout(self.G, seed=42)
        plt.figure(figsize=(12, 9))
        node_colors = []
        for n in self.G.nodes():
            t = self.G.nodes[n].get('type')
            if t == 'user': node_colors.append('skyblue')
            elif t == 'movie': node_colors.append('lightgreen')
            elif t == 'genre': node_colors.append('orange')
            else: node_colors.append('pink')
        nx.draw(self.G, pos, with_labels=True, node_color=node_colors, node_size=900)
        plt.title("Full Graph: Users → Movies → Genres")
        plt.show()

    def plot_recommendations(self, user, K=5):
        recs = self.recommend_best_first(user, K=K)
        sub = self.G.subgraph(recs + [user])
        pos = nx.spring_layout(sub, seed=42)
        colors = []
        for n in sub.nodes():
            if n == user:
                colors.append('skyblue')
            elif n in recs:
                colors.append('red')
            else:
                colors.append('lightgray')
        plt.figure(figsize=(8, 6))
        nx.draw(sub, pos, with_labels=True, node_color=colors, node_size=1000)
        plt.title(f"Top-{K} Recommendations for {user}")
        plt.show()


# =====================================================
# 🎨 TKINTER GUI INTERFACE
# =====================================================
class MovieRecommenderApp:
    def _init_(self, root):
        self.root = root
        self.root.title("🎬 Graph-Based Movie Recommendation System")
        self.root.geometry("700x500")
        self.root.configure(bg="#e3f2fd")

        self.recommender = MovieRecommender()

        title_label = tk.Label(root, text="🎥 Movie Recommendation System",
                               font=("Helvetica", 18, "bold"), bg="#e3f2fd", fg="#0d47a1")
        title_label.pack(pady=10)

        self.tab_control = ttk.Notebook(root)
        self.tab_control.pack(expand=1, fill="both")

        self.tab1 = ttk.Frame(self.tab_control)
        self.tab2 = ttk.Frame(self.tab_control)
        self.tab3 = ttk.Frame(self.tab_control)
        self.tab4 = ttk.Frame(self.tab_control)

        self.tab_control.add(self.tab1, text="➕ Add Data")
        self.tab_control.add(self.tab2, text="⭐ Recommend")
        self.tab_control.add(self.tab3, text="📊 Visualize Graph")
        self.tab_control.add(self.tab4, text="ℹ About")

        self.build_tab1()
        self.build_tab2()
        self.build_tab3()
        self.build_tab4()

    # ----------------- TAB 1 -----------------
    def build_tab1(self):
        tk.Label(self.tab1, text="User ID:").grid(row=0, column=0, padx=10, pady=10)
        self.user_entry = tk.Entry(self.tab1)
        self.user_entry.grid(row=0, column=1)

        tk.Label(self.tab1, text="Movie ID:").grid(row=1, column=0, padx=10, pady=10)
        self.movie_entry = tk.Entry(self.tab1)
        self.movie_entry.grid(row=1, column=1)

        tk.Label(self.tab1, text="Rating (1–5):").grid(row=2, column=0, padx=10, pady=10)
        self.rating_entry = tk.Entry(self.tab1)
        self.rating_entry.grid(row=2, column=1)

        tk.Label(self.tab1, text="Genre:").grid(row=3, column=0, padx=10, pady=10)
        self.genre_entry = tk.Entry(self.tab1)
        self.genre_entry.grid(row=3, column=1)

        tk.Button(self.tab1, text="Add Rating", bg="#64b5f6", fg="white",
                  command=self.add_rating).grid(row=4, column=0, columnspan=2, pady=15)

    def add_rating(self):
        user = self.user_entry.get().strip()
        movie = self.movie_entry.get().strip()
        rating = self.rating_entry.get().strip()
        genre = self.genre_entry.get().strip()

        if not (user and movie and rating):
            messagebox.showwarning("Missing Data", "Please fill all required fields.")
            return
        try:
            r = float(rating)
            if not (1 <= r <= 5):
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Rating must be between 1 and 5.")
            return

        self.recommender.add_rating(user, movie, r)
        if genre:
            self.recommender.link_movie_genre(movie, genre)
        messagebox.showinfo("Success", f"Added rating for {movie} by {user}!")

    # ----------------- TAB 2 -----------------
    def build_tab2(self):
        tk.Label(self.tab2, text="Enter User ID:").pack(pady=10)
        self.rec_user_entry = tk.Entry(self.tab2)
        self.rec_user_entry.pack(pady=5)

        tk.Label(self.tab2, text="Number of Recommendations (K):").pack(pady=10)
        self.k_entry = tk.Entry(self.tab2)
        self.k_entry.insert(0, "5")
        self.k_entry.pack(pady=5)

        tk.Button(self.tab2, text="Get Recommendations", bg="#43a047", fg="white",
                  command=self.show_recommendations).pack(pady=15)

        self.output_text = tk.Text(self.tab2, height=10, width=70, wrap="word", state="disabled")
        self.output_text.pack(padx=10, pady=10)

    def show_recommendations(self):
        user = self.rec_user_entry.get().strip()
        k_val = self.k_entry.get().strip()
        if not user:
            messagebox.showwarning("Input Needed", "Please enter a user ID.")
            return
        try:
            K = int(k_val)
        except ValueError:
            messagebox.showerror("Invalid Input", "K must be an integer.")
            return

        recs = self.recommender.recommend_best_first(user, K=K)
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", tk.END)
        if not recs:
            self.output_text.insert(tk.END, f"No recommendations found for user {user}.\n")
        else:
            self.output_text.insert(tk.END, f"🎬 Top-{K} Recommended Movies for {user}:\n\n")
            for i, m in enumerate(recs, 1):
                self.output_text.insert(tk.END, f"{i}. {m}\n")
        self.output_text.config(state="disabled")

    # ----------------- TAB 3 -----------------
    def build_tab3(self):
        tk.Label(self.tab3, text="Visualization Options:", font=("Helvetica", 12, "bold")).pack(pady=10)
        tk.Button(self.tab3, text="Show Full Graph", bg="#0288d1", fg="white",
                  command=self.recommender.plot_full_graph).pack(pady=10)
        tk.Button(self.tab3, text="Show Recommendations Graph", bg="#c2185b", fg="white",
                  command=self.plot_rec_graph).pack(pady=10)

    def plot_rec_graph(self):
        user = simpledialog.askstring("Input", "Enter User ID:")
        if user:
            self.recommender.plot_recommendations(user)

    # ----------------- TAB 4 -----------------
    def build_tab4(self):
        text = (
            "📘 Graph-Based Movie Recommender System\n\n"
            "• Combines collaborative & content-based filtering.\n"
            "• Models users, movies & genres as graph nodes.\n"
            "• Uses BFS & weighted scoring for recommendations.\n"
            "• Visualizes relationships with NetworkX + Matplotlib.\n\n"
            "Developed with Python & Tkinter."
        )
        tk.Label(self.tab4, text=text, justify="left", wraplength=600,
                 bg="#e3f2fd").pack(padx=15, pady=20)


# =====================================================
# 🚀 RUN APPLICATION
# =====================================================
if _name_ == "_main_":
    root = tk.Tk()
    app = MovieRecommenderApp(root)
    root.mainloop()
