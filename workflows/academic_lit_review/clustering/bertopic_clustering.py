"""BERTopic statistical clustering implementation."""

import logging
from typing import Any

from workflows.academic_lit_review.state import BERTopicCluster

from .constants import MIN_CLUSTER_SIZE

logger = logging.getLogger(__name__)


async def run_bertopic_clustering_node(state: dict) -> dict[str, Any]:
    """Statistical clustering using BERTopic.

    Process:
    1. Create document representations from paper summaries
    2. Embed documents using sentence-transformers
    3. Reduce dimensionality with UMAP
    4. Cluster with HDBSCAN
    5. Extract topic representations
    """
    document_texts = state.get("document_texts", [])
    document_dois = state.get("document_dois", [])

    if len(document_texts) < MIN_CLUSTER_SIZE:
        logger.warning(
            f"Too few documents for BERTopic clustering: {len(document_texts)}"
        )
        return {
            "bertopic_clusters": [],
            "bertopic_error": "Too few documents for statistical clustering",
        }

    try:
        from bertopic import BERTopic
        from sklearn.feature_extraction.text import CountVectorizer

        # Configure CountVectorizer with stop word removal and minimum document frequency
        # This prevents common words like "the", "and" from dominating topic representations
        vectorizer_model = CountVectorizer(
            stop_words="english",
            min_df=2,  # Word must appear in at least 2 documents
            ngram_range=(1, 2),  # Include bigrams for better topic representation
        )

        # Configure BERTopic with improved settings for academic corpora
        topic_model = BERTopic(
            embedding_model="all-MiniLM-L6-v2",
            vectorizer_model=vectorizer_model,
            min_topic_size=MIN_CLUSTER_SIZE,
            nr_topics="auto",  # Let it determine optimal number
            calculate_probabilities=True,
            verbose=False,
        )

        # Fit and transform
        topics, probs = topic_model.fit_transform(document_texts)

        # Build cluster output
        clusters: list[BERTopicCluster] = []
        topic_ids = set(topics)

        for topic_id in topic_ids:
            if topic_id == -1:  # Skip outliers
                continue

            # Get papers in this cluster
            cluster_dois = [
                document_dois[i] for i, t in enumerate(topics) if t == topic_id
            ]

            if not cluster_dois:
                continue

            # Get topic representation (top words)
            topic_info = topic_model.get_topic(topic_id)
            topic_words = [word for word, _ in topic_info[:10]] if topic_info else []

            # Calculate average probability for coherence score
            cluster_indices = [i for i, t in enumerate(topics) if t == topic_id]
            coherence = float(probs[cluster_indices].mean()) if len(cluster_indices) > 0 else 0.0

            clusters.append(
                BERTopicCluster(
                    cluster_id=int(topic_id),
                    topic_words=topic_words,
                    paper_dois=cluster_dois,
                    coherence_score=coherence,
                )
            )

        # Handle outlier papers (topic_id == -1)
        outlier_dois = [
            document_dois[i] for i, t in enumerate(topics) if t == -1
        ]
        if outlier_dois:
            logger.info(f"BERTopic: {len(outlier_dois)} papers not assigned to clusters")

        # Log detailed cluster information for diagnostics
        logger.info(
            f"BERTopic clustering complete: {len(clusters)} clusters "
            f"from {len(document_texts)} documents"
        )
        for c in clusters[:5]:  # Log first 5 clusters for diagnostics
            logger.info(
                f"  Cluster {c['cluster_id']}: {len(c['paper_dois'])} papers, "
                f"topics: {c['topic_words'][:5]}"
            )

        return {
            "bertopic_clusters": clusters,
            "bertopic_error": None,
        }

    except ImportError:
        error_msg = "BERTopic not installed. Install with: pip install bertopic"
        logger.error(error_msg)
        return {
            "bertopic_clusters": [],
            "bertopic_error": error_msg,
        }
    except Exception as e:
        error_msg = f"BERTopic clustering failed: {str(e)}"
        logger.error(error_msg)
        return {
            "bertopic_clusters": [],
            "bertopic_error": error_msg,
        }


def _evaluate_bertopic_quality(
    bertopic_clusters: list[BERTopicCluster],
    total_papers: int,
) -> tuple[bool, str]:
    """Evaluate quality of BERTopic clustering results.

    Returns:
        (is_good_quality, reason) tuple
    """
    if not bertopic_clusters:
        return False, "No clusters produced"

    # Check 1: Too few clusters for the corpus size
    # For 50+ papers, we expect at least 4-5 clusters
    expected_min_clusters = max(3, total_papers // 15)
    if len(bertopic_clusters) < expected_min_clusters:
        return False, f"Too few clusters ({len(bertopic_clusters)}) for {total_papers} papers"

    # Check 2: Cluster imbalance - one cluster dominates
    cluster_sizes = [len(c["paper_dois"]) for c in bertopic_clusters]
    if cluster_sizes:
        max_size = max(cluster_sizes)
        if max_size > total_papers * 0.6:  # One cluster has >60% of papers
            return False, f"Cluster imbalance: largest cluster has {max_size}/{total_papers} papers"

    # Check 3: Topic words quality - check for stop words or overly generic terms
    # Common stop words and domain-generic terms that indicate poor clustering
    bad_topic_words = {
        "the", "and", "to", "of", "in", "for", "a", "is", "that", "with",
        "on", "are", "be", "this", "an", "as", "it", "by", "from", "or",
        "using", "based", "can", "we", "our", "their", "these", "which",
    }

    for cluster in bertopic_clusters:
        topic_words = cluster.get("topic_words", [])[:5]  # Check top 5 words
        bad_count = sum(1 for w in topic_words if w.lower() in bad_topic_words)
        if bad_count >= 3:  # More than half are bad
            return False, f"Poor topic words in cluster {cluster['cluster_id']}: {topic_words}"

    return True, "BERTopic quality acceptable"
