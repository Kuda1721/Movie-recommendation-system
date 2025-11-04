Core Concepts Used
Concept	Where It Appears	Purpose
Graph Theory	networkx graph self.G	Represent users, movies, genres, and tags as connected nodes
Collaborative Filtering	score_movie() (user–movie ratings)	Learn from similar users’ ratings
Content-Based Filtering	score_movie() (genre + tag overlap)	Match user’s past interests by genre/tags
Hybrid Recommender System	Combined scoring model	Blends multiple techniques for better accuracy
BFS (Breadth-First Search)	bfs_candidates()	Find movies “close” to user in the graph
Weighted Scoring	weights dict	Combine similarity, genre, tag, popularity, bias
Heap/Priority Queue	recommend_best_first()	Rank and pick top-K best movies efficiently
Data Visualization	plot_* methods	Show graph relationships visually
Data Normalization	score_movie()	Normalize ratings and popularity
Machine Learning Preparation	train_test_split imported	Ready for model testing & evaluation
