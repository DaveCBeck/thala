"""Clustering configuration constants."""

MIN_CLUSTER_SIZE = 3  # Minimum papers per cluster
MAX_CLUSTERS = 15  # Maximum number of final clusters
MIN_CLUSTERS = 5  # Minimum number of final clusters
MIN_PAPERS_FOR_BERTOPIC = 20  # Skip BERTopic for small corpora, use LLM clustering only
