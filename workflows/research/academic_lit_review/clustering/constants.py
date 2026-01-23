"""Clustering configuration constants."""

MIN_CLUSTER_SIZE = 3  # Minimum papers per cluster
MAX_CLUSTERS = 6  # Maximum number of final clusters (broad themes)
MIN_CLUSTERS = 3  # Minimum number of final clusters (allows flexibility for smaller corpora)
MIN_PAPERS_FOR_BERTOPIC = 20  # Skip BERTopic for small corpora, use LLM clustering only
