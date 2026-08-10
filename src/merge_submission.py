"""
Merges prediction.json and response.json in-place

Arguments:
  -p, --preds     [Required] Path to prediction.json file to update.
  -r, --response  [Required] Direct path to  response.json file.

Usage:
  uv run merge_submission.py -p path/to/prediction.json -r path/to/response.json
"""

from typing import List, Sequence
from pathlib import Path
import argparse
import json

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

def main(
    preds_path: Path,
    response_path: Path
):
    print(f"Track file (will be updated in-place): {preds_path}")
    print(f"Using response file: {response_path}")

    if not preds_path.exists():
        print(f"Error: Track file {preds_path} not found.")
        return

    if not response_path.exists():
        print(f"Error: Response file {response_path} not found.")
        return

    # Load data from both files
    with preds_path.open() as f:
        submission: list = json.load(f)

    with response_path.open() as f:
        response_data: dict = json.load(f)

    merged_data = []
    for item in submission:
        session_id = item.get("session_id")
        if session_id in response_data:
            item["predicted_response"] = response_data[session_id]
            merged_data.append(item)
        
        else:
            raise ValueError(f"Session ID {session_id} not found in response data.")

    print(f"Total entries in {preds_path}: {len(submission)}")
    print(len(merged_data), "entries matched between prediction and response data.")

    assert len(merged_data) == len(submission), "Matched data length does not match submission data length."
    assert len(merged_data) == 80, "Matched data length does not match expected length of 80."

    # Compute lexical diversity (Distinct-2) for the merged responses
    all_responses = [item["predicted_response"] for item in merged_data]
    distinct_score = compute_lexical_diversity(all_responses, n=2)
    print(f"\n---> Estimated Lexical Diversity (Distinct-2): {distinct_score:.4f} <---\n")

    # Save merged data in-place to the track file
    with preds_path.open("w") as f:
        json.dump(merged_data, f, indent=4)

    print(f"Successfully updated track data in-place at {preds_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge prediction.json and response.json in-place.")
    parser.add_argument("-p", "--preds", dest="preds_path", required=True, help="Path to the predictions file to update")
    parser.add_argument("-r", "--response", dest="response_path", required=True, help="Path to the response file to merge")
    
    args = parser.parse_args()

    main(
        preds_path=Path(args.preds_path),
        response_path=Path(args.response_path)
    )