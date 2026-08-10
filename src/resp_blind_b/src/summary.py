from typing import List, Dict
from pathlib import Path

from llama_cpp import Llama

from src.utils.resources import SUMMARIES_CACHE_DIR

# In-memory cache to avoid redundant file reads during a single run
cached_summaries = None  

SYSTEM_PROMPT = """You are a strict data-extraction analyst for a recommendation system. 
Extract a concise bulleted list of the user's explicit preferences, negative preferences (dislikes), and hard constraints from the chat history. 
Output ONLY the bullet points. Do not include any introductory or concluding text."""

def generate_summary(
    client: Llama,
    conversation: List[Dict[str, str]],
    **kwargs
) -> str:
        # Format the raw history text for the summarizer to read
        formatted_history = ""
        for turn in conversation:
            role = turn['role'].upper()
            content = turn['content']
            formatted_history += f"{role}: {content}\n"
        
        # Build the summarization chat template
        summarizer_messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"Extract the user profile from this history:\n\n<history>\n{formatted_history}\n</history>"
            }
        ]
        
        # Generate summary 
        response = client.create_chat_completion(
            messages=summarizer_messages,
            **kwargs
        )
        
        if response["choices"][0]["finish_reason"] == "length":
            print("Warning: Summary may be truncated. Consider increasing max_tokens.")
        
        return response["choices"][0]["message"]["content"]

def load_summaries_cache() -> Dict[str, str]:
    global cached_summaries
    if cached_summaries is not None: return
    SUMMARIES_CACHE_DIR.mkdir(exist_ok=True)
    
    cached_summaries = {}
    for file in SUMMARIES_CACHE_DIR.glob("*.txt"):
        session_id = file.stem
        with open(file, "r", encoding="utf-8") as f:
            cached_summaries[session_id] = f.read()


def get_summary(
    session_id: str,
    client: Llama = None,
    conversation: List[Dict[str, str]] | None = None,
    fetch_from_cache: bool = True,
    output_dir: Path | None = None,
    **kwargs
) -> str:
    # Load the cache if we want to fetch from it
    if fetch_from_cache:
        load_summaries_cache()  
        
        # Check if the session is in it
        if session_id in cached_summaries:
            return cached_summaries[session_id]
    
    # If not in cache (or not fetching from cache), proceed to generate
    if conversation is None:
        raise ValueError("No conversation provided to generate summary, and summary not found in cache.")
    
    elif client is None:
        raise ValueError("No LLM client provided to generate summary, and summary not found in cache.")

    else:
        summary = "No previous history."
        if len(conversation) > 0:
            summary = generate_summary(client, conversation, **kwargs)
        
        if output_dir is not None:
            # Ensure the output directory exists
            output_path = output_dir / "summaries"
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save the summary to disk for future runs
            summary_file = output_path / f"{session_id}.txt"
            with open(summary_file, "w", encoding="utf-8") as f:
                f.write(summary)
        
        else:
            # Cache the summary to disk for future runs
            summary_file = SUMMARIES_CACHE_DIR / f"{session_id}.txt"
            with open(summary_file, "w", encoding="utf-8") as f:
                f.write(summary)

        return summary