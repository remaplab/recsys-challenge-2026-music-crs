from typing import List, Sequence, Dict

def _whitespace_tokens(text: str) -> List[str]:
    """Tokenize with whitespace split only (no normalization)."""
    return (text or "").split()

def compute_lexical_diversity(list_of_responses: Sequence[str], n: int = 2) -> float:
    """
    Lexical diversity with Distinct-2.
    """
    ngrams = set()
    total_ngrams = 0

    for response in list_of_responses:
        tokens = _whitespace_tokens(response.lower())
        if len(tokens) < n:
            continue

        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams.add(ngram)
            total_ngrams += 1

    if total_ngrams == 0:
        return 0.0

    return len(ngrams) / float(total_ngrams)


def get_bigrams(text: str) -> List[tuple]:
    tokens = _whitespace_tokens(text.lower())
    if len(tokens) < 2:
        return []
    return [tuple(tokens[i:i+2]) for i in range(len(tokens) - 1)]

def get_count_dict(text: str) -> Dict[tuple, int]:
    bigrams = get_bigrams(text)
    return {bigram: bigrams.count(bigram) for bigram in set(bigrams)}

def update_count_dict(count_dict: Dict[tuple, int], from_text: str, to_text: str) -> Dict[tuple, int]:
    from_bigrams = get_bigrams(from_text)
    to_bigrams = get_bigrams(to_text)

    for bigram in from_bigrams:
        if bigram in count_dict:
            count_dict[bigram] -= 1
            if count_dict[bigram] <= 0:
                del count_dict[bigram]

    for bigram in to_bigrams:
        count_dict[bigram] = count_dict.get(bigram, 0) + 1

    return count_dict