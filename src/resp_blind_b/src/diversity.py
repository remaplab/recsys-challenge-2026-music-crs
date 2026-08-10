import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Set, Tuple

def get_bigrams(text: str) -> Set[Tuple[str, str]]:
    """Extracts a set of bigrams from a lowercase string."""
    tokens = text.lower().split()
    return set(zip(tokens, tokens[1:]))

def jaccard_similarity(set1: set, set2: set) -> float:
    """Calculates intersection over union for two sets."""
    if not set1 and not set2: 
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

def get_most_overlapping_pair(responses: List[dict]) -> Tuple[dict, dict]:
    """Finds the pair of responses that share the most bigrams."""
    bigram_sets = [get_bigrams(r['text']) for r in responses]
    
    max_similarity = -1
    best_pair = (None, None)
    
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            sim = jaccard_similarity(bigram_sets[i], bigram_sets[j])
            if sim > max_similarity:
                max_similarity = sim
                best_pair = (i, j)
                
    return best_pair

def get_most_overlapping_batch(responses: List[dict], batch_size: int = 5) -> Tuple[List[dict], List[int]]:
    """
    Scans the entire pool and extracts the single cluster of `batch_size` 
    that shares the most bigrams.
    """
    texts = [r['text'] for r in responses]
    vectorizer = CountVectorizer(ngram_range=(2, 2), analyzer='word')
    X = vectorizer.fit_transform(texts)
    
    sim_matrix = cosine_similarity(X)
    # Zero out the diagonal so we don't match an item with itself
    np.fill_diagonal(sim_matrix, 0)
    
    # 1. Find the absolute highest overlapping pair in the whole dataset
    max_idx = np.argmax(sim_matrix)
    idx1, idx2 = np.unravel_index(max_idx, sim_matrix.shape)
    
    batch_indices = [idx1, idx2]
    
    # 2. Greedily add the next items most similar to this cluster
    unbatched = set(range(len(responses))) - set(batch_indices)
    
    while len(batch_indices) < batch_size and unbatched:
        best_k = None
        max_sim_sum = -1
        
        for k in unbatched:
            # Calculate how similar item 'k' is to the items currently in our bad batch
            sim_sum = sum(sim_matrix[k, b] for b in batch_indices)
            if sim_sum > max_sim_sum:
                max_sim_sum = sim_sum
                best_k = k
                
        batch_indices.append(best_k)
        unbatched.remove(best_k)
        
    return [responses[i] for i in batch_indices], batch_indices