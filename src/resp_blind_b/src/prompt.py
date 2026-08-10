from typing import Dict
from llama_cpp import Llama
from src.summary import get_summary  

def build_prompt(
    client: Llama,
    row: Dict,
    system_prompt: str,
    turns_to_keep: int = 0,
    no_history_summary: bool = False,
    **kwargs
):
    session_id = row.get("session_id")

    # Split Conversation into History and Final Query
    conversations = row.get("conversations", [])
    if not conversations:
        raise ValueError("No conversation data found.")
    
    assert conversations[-1].get("role") == "user", "The last turn in the conversation must be from the user."
    final_query = conversations[-1].get("content")

    # Filter out role music (uuid not easy to parse) and keep only the last N turns of actual conversation
    conversations = [turn for turn in conversations if turn.get("role") in ["user", "assistant"]]

    # Split the conversation into history and recent turns to keep
    recent_turns = conversations[-turns_to_keep-1:-1] if turns_to_keep > 0 else []
    history = conversations[:-turns_to_keep-1] if turns_to_keep > 0 else conversations[:-1]
    
    # Generate Summary if there is history
    summary_text = "No previous history."
    if history and not no_history_summary:
        summary_text = get_summary(
            session_id=session_id,
            client=client,
            conversation=history,
            fetch_from_cache=True,
            **kwargs
        )

    # Insert the generated summary into the system prompt                
    final_system_prompt = system_prompt.replace("{generated_summary_from_prompt_1}", summary_text)

    # Assemble chat template for the response generator
    chat_template = []
    chat_template.append({"role": "system", "content": final_system_prompt})
    for turn in recent_turns:
        chat_template.append({"role": turn.get("role"), "content": turn.get("content")})
    chat_template.append({"role": "user", "content": final_query})

    return chat_template