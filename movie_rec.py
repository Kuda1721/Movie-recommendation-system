# movie_recommender.py
# Requires: networkx, numpy, scikit-learn
# pip install networkx numpy scikit-learn

import networkx as nx
from collections import deque, defaultdict
import heapq
import random
import math
import numpy as np
from sklearn.model_selection import train_test_split

class MovieRecommender:
    """
    MovieRecommender - builds a user/movie/genre/tag graph and provides multiple
    recommendation/search algorithms, evaluation, tuning, and export-to-Neo4j helpers.
    """
    def __init__(self):
        self.G = nx.Graph()
        self.users = set()
        self.movies = set()
        self.genres = set()
        self.tags = set()
        # store ratings list (user, movie, rating) for easy splitting/eval
        self.ratings = []

    # --------------------------
    # Graph building utilities
    # --------------------------
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
        # rating numeric (e.g., 1-5); store edge with rating and normalized weight
        self.add_user(user); self.add_movie(movie)
        self.G.add_edge(user, movie, type='rated', rating=float(rating), weight=float(rating)/5.0)
        self.ratings.append((user, movie, float(rating)))

    def add_like(self, user, movie):
        self.add_user(user); self.add_movie(movie)
        self.G.add_edge(user, movie, type='like')

    def link_movie_genre(self, movie, genre):
        self.add_movie(movie); self.add_genre(genre)
        self.G.add_edge(movie, genre, type='genre')

    def link_movie_tag(self, movie, tag):
        self.add_movie(movie); self.add_tag(tag)
        self.G.add_edge(movie, tag, type='tag')

    # --------------------------
    # Graph query helpers
    # --------------------------
    def user_rated_movies(self, user):
        return [n for n in self.G.neighbors(user) if self.G.nodes[n].get('type')=='movie']

    def movie_users(self, movie):
        return [n for n in self.G.neighbors(movie) if self.G.nodes[n].get('type')=='user']

    # --------------------------
    # BFS / DFS candidate generation
    # --------------------------
    def bfs_candidates(self, user, max_depth=3):
        """
        BFS out from user and return candidate movies not already rated/liked by user.
        We count the number of distinct (user→movie) paths (simple walk-based scoring).
        More efficient than naive operations by avoiding revisiting nodes.
        """
        if user not in self.G: return {}
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
                q.append((nbr, depth+1))
                if self.G.nodes[nbr].get('type') == 'movie':
                    # only consider movies user hasn't connected to
                    if not self.G.has_edge(user, nbr):
                        # weight candidate by inverse depth (closer nodes count more)
                        candidates[nbr] += 1.0 / (depth + 1)
        return dict(candidates)

    def dfs_collect(self, start, max_nodes=100):
        """
        Simple DFS collector for exploratory analysis.
        """
        visited = set()
        stack = [start]
        collected = []
        while stack and len(collected) < max_nodes:
            node = stack.pop()
            if node in visited: continue
            visited.add(node)
            collected.append(node)
            for nbr in self.G.neighbors(node):
                if nbr not in visited:
                    stack.append(nbr)
        return collected

    # --------------------------
    # Scoring: content + collaborative
    # --------------------------
    def score_movie(self, user, movie, weights):
        """
        Score a candidate movie for user using a combination of:
         - collaborative similarity (avg neighbor rating normalized)
         - genre overlap (fraction)
         - tag overlap (fraction)
         - popularity (user-degree normalized)
         - personal bias (if user previously liked this genre/tag)
        weights: dict with keys w_sim, w_genre, w_tag, w_pop, w_bias
        """
        # 1) collaborative similarity: average rating by other users for this movie
        neigh_users = [n for n in self.G.neighbors(movie) if self.G.nodes[n].get('type')=='user' and n!=user]
        sim_score = 0.0; count=0
        for u in neigh_users:
            if self.G.has_edge(u, movie):
                r = self.G[u][movie].get('rating', None)
                if r is not None:
                    sim_score += r
                    count += 1
        sim_norm = (sim_score / (5.0*count)) if count>0 else 0.0

        # 2) genre overlap between user's rated movies and movie
        user_genres = set()
        for m in self.user_rated_movies(user):
            for g in self.G.neighbors(m):
                if self.G.nodes[g].get('type')=='genre':
                    user_genres.add(g)
        movie_genres = {g for g in self.G.neighbors(movie) if self.G.nodes[g].get('type')=='genre'}
        genre_overlap = (len(user_genres & movie_genres) / (len(user_genres) + 1e-9)) if user_genres else 0.0

        # 3) tag overlap (finer-grained content)
        user_tags = set()
        for m in self.user_rated_movies(user):
            for t in self.G.neighbors(m):
                if self.G.nodes[t].get('type')=='tag':
                    user_tags.add(t)
        movie_tags = {t for t in self.G.neighbors(movie) if self.G.nodes[t].get('type')=='tag'}
        tag_overlap = (len(user_tags & movie_tags) / (len(user_tags) + 1e-9)) if user_tags else 0.0

        # 4) popularity (normalized by sqrt to dampen)
        pop = len(self.movie_users(movie))
        pop_norm = math.sqrt(pop) / math.sqrt(max(1, max(1, len(self.users))))  # safe normalize

        # 5) personal bias: user's avg rating (higher-rated users may prefer higher-rated movies)
        uratings = [self.G[user][m].get('rating') for m in self.user_rated_movies(user) if self.G.has_edge(user,m)]
        user_bias = (sum(uratings)/len(uratings))/5.0 if uratings else 0.0

        s = (weights.get('w_sim',0.5)*sim_norm +
             weights.get('w_genre',0.2)*genre_overlap +
             weights.get('w_tag',0.05)*tag_overlap +
             weights.get('w_pop',0.15)*pop_norm +
             weights.get('w_bias',0.1)*user_bias)
        return s

    # --------------------------
    # Best-First recommendation (uses BFS candidates + scoring)
    # --------------------------
    def recommend_best_first(self, user, K=5, weights=None, max_depth=3):
        if weights is None:
            weights = {'w_sim':0.55,'w_genre':0.25,'w_tag':0.05,'w_pop':0.1,'w_bias':0.05}
        cand = self.bfs_candidates(user, max_depth=max_depth)
        heap = []
        for m, base in cand.items():
            sc = self.score_movie(user, m, weights)
            heapq.heappush(heap, (-sc, m))
        recs=[]
        while heap and len(recs)<K:
            recs.append(heapq.heappop(heap)[1])
        return recs

    # --------------------------
    # A* search: treat paths from user->...->movie and use heuristic to prioritize
    # --------------------------
    def a_star_recommend(self, user, K=5, weights=None, max_expansions=500):
        """
        A* variant where:
         - g(n): negative accumulated score along path (we want high score -> low cost)
         - h(n): heuristic estimate for best possible additional score (admissible)
        We search nodes and when we pop a movie node, we add to recommendations.
        This is a heuristic-guided best-first search (A* inspired).
        """
        if user not in self.G:
            return []
        if weights is None:
            weights = {'w_sim':0.55,'w_genre':0.25,'w_tag':0.05,'w_pop':0.1,'w_bias':0.05}

        def heuristic(node):
            # If node is movie, heuristic is 0 (we're at a target)
            if self.G.nodes[node].get('type')=='movie':
                return 0.0
            # Estimate by looking at neighbors that are movies: take max possible score
            best = 0.0
            for nbr in self.G.neighbors(node):
                if self.G.nodes[nbr].get('type')=='movie':
                    best = max(best, self.score_movie(user,nbr,weights))
            # h must be a cost (we minimize), so we return negative of estimated reward
            return -best

        # open: priority queue of (f = g + h, g, node, path)
        # start at user with g=0
        open_heap = []
        start = user
        start_g = 0.0
        start_h = heuristic(start)
        heapq.heappush(open_heap, (start_g + start_h, start_g, start, [start]))
        closed = set()
        recs = []
        expansions = 0
        while open_heap and len(recs)<K and expansions < max_expansions:
            f, g, node, path = heapq.heappop(open_heap)
            if (node, g) in closed:
                continue
            closed.add((node,g))
            expansions += 1
            if self.G.nodes[node].get('type')=='movie' and not self.G.has_edge(user,node):
                # found a candidate movie
                recs.append(node)
                continue
            # expand neighbors
            for nbr in self.G.neighbors(node):
                if nbr in path:  # avoid immediate cycles
                    continue
                # cost: we use negative incremental score when moving to a movie
                delta = 0.0
                if self.G.nodes[nbr].get('type')=='movie':
                    delta = - self.score_movie(user, nbr, weights)
                # else delta 0 for intermediate nodes
                new_g = g + delta
                new_h = heuristic(nbr)
                heapq.heappush(open_heap, (new_g + new_h, new_g, nbr, path + [nbr]))
        return recs

    # --------------------------
    # Simple Minimax simulation for two competing platforms
    # --------------------------
    def simulate_platform_competition(self, user, platform_movies_A, platform_movies_B, K=3, weights=None):
        """
        Two platforms choose top-K movies to promote to the user.
        We model utility of platform = sum of predicted scores for the movies the user chooses.
        Minimax: Platform A chooses a set to maximize its minimum utility assuming B responds optimally.
        For simplicity we enumerate small candidate sets (cartesian of top candidates).
        """
        # build candidate set (top 10 best-first)
        if weights is None:
            weights = {'w_sim':0.55,'w_genre':0.25,'w_tag':0.05,'w_pop':0.1,'w_bias':0.05}
        cand_scores = []
        candidates = self.bfs_candidates(user, max_depth=3)
        for m in candidates:
            cand_scores.append((self.score_movie(user,m,weights), m))
        cand_scores.sort(reverse=True)
        top_candidates = [m for _,m in cand_scores[:10]]
        # if given platform movie pools, prefer them
        poolA = [m for m in top_candidates if m in platform_movies_A] or top_candidates[:5]
        poolB = [m for m in top_candidates if m in platform_movies_B] or top_candidates[:5]

        # enumerate small subsets (combinatorial explosion avoided by limiting)
        from itertools import combinations
        bestA = None; bestA_utility = -1e9
        # For each A choice, assume B chooses best response to minimize A's utility (i.e., worst-case for A)
        for A_choice in combinations(poolA, min(K,len(poolA))):
            # compute A utility if user picks movies proportional to score
            A_scores = [self.score_movie(user,m,weights) for m in A_choice]
            utilA = sum(A_scores)
            # B's possible choices
            worstA = 1e9
            for B_choice in combinations(poolB, min(K,len(poolB))):
                # if B_choice overlaps heavily with A_choice it can reduce A's exclusivity -> we simulate as reducing A's utility
                overlap = len(set(A_choice) & set(B_choice))
                # simple adversarial effect: decrease A utility proportional to overlap
                adj_utilA = utilA * (1.0 - 0.25*overlap)
                if adj_utilA < worstA:
                    worstA = adj_utilA
            # A wants to maximize its worst-case (minimax)
            if worstA > bestA_utility:
                bestA_utility = worstA
                bestA = A_choice
        return {'best_A_choice': bestA, 'worst_case_utility': bestA_utility}

    # --------------------------
    # Evaluation: Precision@K and NDCG@K using holdout split
    # --------------------------
    def train_test_split_holdout(self, test_size=0.2, random_state=42):
        """
        Split stored ratings into train/test by user: we ensure each user in test has at least
        one rating in train as well (simple holdout stratified by user).
        This function builds two separate graphs (train_graph and test_list)
        """
        if not self.ratings:
            return None, None, None
        # per-user holdout: for each user, hold out at most one rating into test
        train = []
        test = []
        by_user = defaultdict(list)
        for u,m,r in self.ratings:
            by_user[u].append((u,m,r))
        rng = random.Random(random_state)
        for u, lst in by_user.items():
            if len(lst) == 1:
                train.extend(lst)
            else:
                # pick one to test
                idx = rng.randrange(len(lst))
                for i,entry in enumerate(lst):
                    if i==idx:
                        test.append(entry)
                    else:
                        train.append(entry)
        # Build train graph copy (so we can evaluate on test)
        train_rec = MovieRecommender()
        # Add same movies/genres/tags metadata to train_rec
        # copy nodes and edges except user->movie rated edges only from train
        for n, d in self.G.nodes(data=True):
            if d.get('type')=='user':
                train_rec.add_user(n)
            elif d.get('type')=='movie':
                train_rec.add_movie(n, meta={k:v for k,v in d.items() if k!='type'})
            elif d.get('type')=='genre':
                train_rec.add_genre(n)
            elif d.get('type')=='tag':
                train_rec.add_tag(n)
        # copy non-rating edges (genre/tag)
        for u,v,data in self.G.edges(data=True):
            t = data.get('type')
            if t in ('genre','tag'):
                if self.G.nodes[u].get('type')=='movie' and self.G.nodes[v].get('type') in ('genre','tag'):
                    if data['type']=='genre':
                        train_rec.link_movie_genre(u,v)
                    else:
                        train_rec.link_movie_tag(u,v)
                elif self.G.nodes[v].get('type')=='movie' and self.G.nodes[u].get('type') in ('genre','tag'):
                    if data['type']=='genre':
                        train_rec.link_movie_genre(v,u)
                    else:
                        train_rec.link_movie_tag(v,u)
        # add rating edges from train
        train_rec.ratings = list(train)
        for u,m,r in train:
            train_rec.add_rating(u,m,r)
        return train_rec, train, test

    @staticmethod
    def precision_at_k(recommended, actual_set, k):
        rec_k = recommended[:k]
        hits = len([r for r in rec_k if r in actual_set])
        return hits / float(k)

    @staticmethod
    def dcg(recommended, actual_ratings, k):
        dcg_val = 0.0
        for i, m in enumerate(recommended[:k]):
            rel = actual_ratings.get(m, 0.0)  # relevance from test rating (0 if not present)
            # use relevance of rating (0..1)
            dcg_val += (2**rel - 1.0) / math.log2(i+2)
        return dcg_val

    @staticmethod
    def ndcg_at_k(recommended, actual_ratings, k):
        dcg_v = MovieRecommender.dcg(recommended, actual_ratings, k)
        # ideal DCG: sort by true relevance
        ideal_sorted = sorted(actual_ratings.values(), reverse=True)
        ideal_rels = ideal_sorted[:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_rels):
            idcg += (2**rel - 1.0) / math.log2(i+2)
        return dcg_v / idcg if idcg>0 else 0.0

    def evaluate(self, K=5, weights=None):
        """
        Split into train/test, build train graph, run recommend_best_first on each user in test,
        compute avg precision@K and ndcg@K.
        """
        train_rec, train_list, test_list = self.train_test_split_holdout()
        if train_rec is None:
            raise RuntimeError("No ratings to evaluate.")
        # For each user in test_list, gather their actual held-out movie(s)
        test_by_user = defaultdict(dict)
        for u,m,r in test_list:
            test_by_user[u][m] = r/5.0  # normalized relevance

        precisions=[]; ndcgs=[]
        for u, actual_r_dict in test_by_user.items():
            # if user not in train graph, skip (shouldn't happen) or add user with no ratings
            if u not in train_rec.G:
                # user cold-start -> skip
                continue
            recs = train_rec.recommend_best_first(u, K=K, weights=weights)
            precisions.append(self.precision_at_k(recs, set(actual_r_dict.keys()), K))
            ndcgs.append(self.ndcg_at_k(recs, actual_r_dict, K))
        if not precisions:
            return {'precision':0.0, 'ndcg':0.0, 'n':0}
        return {'precision': float(np.mean(precisions)), 'ndcg': float(np.mean(ndcgs)), 'n': len(precisions)}

    # --------------------------
    # Hill-climbing tuning using true evaluation
    # --------------------------
    def hill_climb_tune(self, initial_weights, steps=200, step_size=0.08, K=5):
        """
        Tune weights to maximize ndcg (or precision). We perform a random local search (hill-climb).
        """
        # ensure weights sum to 1
        w = dict(initial_weights)
        s = sum(w.values()) or 1.0
        w = {k: v/s for k,v in w.items()}
        best = dict(w)
        best_score = self.evaluate(K=K, weights=best)['ndcg']
        for i in range(steps):
            cand = {k: max(0.0, min(1.0, best[k] + random.uniform(-step_size, step_size))) for k in best}
            s = sum(cand.values()) or 1.0
            cand = {k: v/s for k,v in cand.items()}
            sc = self.evaluate(K=K, weights=cand)['ndcg']
            if sc > best_score:
                best, best_score = cand, sc
        return best, best_score

    # --------------------------
    # Neo4j / Cypher export helper
    # --------------------------
    def export_to_cypher(self):
        """
        Returns a string containing Cypher commands to create nodes/relationships for Neo4j.
        Useful to copy/paste into Neo4j Browser. (Lightweight: no batching.)
        """
        lines = []
        # create nodes with labels
        for n, d in self.G.nodes(data=True):
            typ = d.get('type')
            if typ == 'user':
                lines.append(f"MERGE (u:User {{id: '{n}'}});")
            elif typ == 'movie':
                props = {k:v for k,v in d.items() if k!='type'}
                props_str = ', '.join([f"{k}: '{v}'" for k,v in props.items()]) if props else ''
                if props_str:
                    lines.append(f"MERGE (m:Movie {{id: '{n}', {props_str}}});")
                else:
                    lines.append(f"MERGE (m:Movie {{id: '{n}'}});")
            elif typ == 'genre':
                lines.append(f"MERGE (g:Genre {{id: '{n}'}});")
            elif typ == 'tag':
                lines.append(f"MERGE (t:Tag {{id: '{n}'}});")
        # relationships
        for u,v,data in self.G.edges(data=True):
            t = data.get('type')
            if t == 'rated':
                r = data.get('rating', 0.0)
                lines.append(f"MATCH (u:User {{id: '{u}'}}),(m:Movie {{id: '{v}'}}) MERGE (u)-[:RATED {{rating: {r}}}]->(m);")
                lines.append(f"MATCH (u:User {{id: '{v}'}}),(m:Movie {{id: '{u}'}}) WHERE false RETURN 1;")  # no-op to keep order safe
            elif t == 'like':
                lines.append(f"MATCH (u:User {{id: '{u}'}}),(m:Movie {{id: '{v}'}}) MERGE (u)-[:LIKED]->(m);")
            elif t == 'genre':
                # ensure movie is left side
                if self.G.nodes[u].get('type')=='movie':
                    lines.append(f"MATCH (m:Movie {{id: '{u}'}}),(g:Genre {{id: '{v}'}}) MERGE (m)-[:IN_GENRE]->(g);")
                else:
                    lines.append(f"MATCH (m:Movie {{id: '{v}'}}),(g:Genre {{id: '{u}'}}) MERGE (m)-[:IN_GENRE]->(g);")
            elif t == 'tag':
                if self.G.nodes[u].get('type')=='movie':
                    lines.append(f"MATCH (m:Movie {{id: '{u}'}}),(t:Tag {{id: '{v}'}}) MERGE (m)-[:HAS_TAG]->(t);")
                else:
                    lines.append(f"MATCH (m:Movie {{id: '{v}'}}),(t:Tag {{id: '{u}'}}) MERGE (m)-[:HAS_TAG]->(t);")
        return "\n".join(lines)

# --------------------------
# Example usage with your sample data
# --------------------------
if __name__ == "__main__":
    rec = MovieRecommender()
    # sample nodes
    users = ['u1','u2','u3']
    movies = ['m1','m2','m3','m4']
    for u in users: rec.add_user(u)
    for m in movies: rec.add_movie(m)
    ratings = [
        ('u1','m1',5), ('u1','m2',3),
        ('u2','m1',4), ('u2','m3',5),
        ('u3','m2',4), ('u3','m3',2), ('u3','m4',5)
    ]
    for u,m,r in ratings:
        rec.add_rating(u,m,r)
    # genres
    rec.link_movie_genre('m1','g_action'); rec.link_movie_genre('m2','g_drama')
    rec.link_movie_genre('m3','g_action'); rec.link_movie_genre('m4','g_action')
    # tags (optional)
    rec.link_movie_tag('m1','t_fast'); rec.link_movie_tag('m4','t_emotional')
    # run best-first
    print("Best-first recs for u1:", rec.recommend_best_first('u1', K=3))
    # run A*
    print("A* recs for u1:", rec.a_star_recommend('u1', K=3))
    # run minimax simulation example (platform pools)
    sim = rec.simulate_platform_competition('u1', platform_movies_A=['m3','m4'], platform_movies_B=['m2','m3'])
    print("Minimax simulation (platform A choice):", sim)
    # evaluate
    print("Evaluation (precision, ndcg):", rec.evaluate(K=2))
    # tuning (might be slow for big graphs, here tiny)
    tuned, score = rec.hill_climb_tune({'w_sim':0.55,'w_genre':0.25,'w_tag':0.05,'w_pop':0.1,'w_bias':0.05}, steps=50)
    print("Tuned weights:", tuned, "score:", score)
    # export cypher snippet
    print("Cypher export (first 10 lines):")
    cy = rec.export_to_cypher().splitlines()
    for line in cy[:10]:
        print(line)