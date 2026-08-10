from pathlib import Path
from typing import Dict, Tuple
import json
import sys

from jinja2 import Template
from llama_cpp import Llama
from tqdm import tqdm
import polars as pl

from src.summary import get_summary

from src.utils.resources import BLIND_B, get_system_prompt, get_template
from src.utils.check import sanity_check
from src.utils.config import parse_config
from src.utils.bigrams import update_count_dict, _whitespace_tokens

def load_jsonl_checkpoint(filepath: Path) -> Dict[str, str]:
    """Helper to load a JSONL checkpoint into a dictionary."""
    data = {}
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    data[obj["session_id"]] = obj["content"]
    return data


def retry_loop(
    client: Llama,
    max_retries: int = 3,
    max_token_increase: int = 50,
    **kwargs
):
    cfg = kwargs.copy()
    trials = 0
    while trials < max_retries:
        try:
            response_payload = client.create_chat_completion(**cfg)
            if response_payload["choices"][0]["finish_reason"] == "length":
                print(f"\nWarning: LLM output was truncated. The response may be incomplete.", file=sys.stderr)
                trials += 1
                cfg["max_tokens"] += max_token_increase
                if trials < max_retries:
                    print("Trying again with increased max_tokens...", file=sys.stderr)
                    continue
                else:
                    print(f"Max retries reached. Saving truncated response and moving on.", file=sys.stderr)
            return response_payload
        
        except Exception as e:
            print(f"\nError during LLM call: {e}", file=sys.stderr)
            trials += 1
            if trials < max_retries:
                print("Retrying...", file=sys.stderr)
            else:
                print(f"Max retries reached. Moving on.", file=sys.stderr)
                raise e


def generate_summaries(
    client: Llama,
    fetch_from_cache: bool = True,
    output_dir: Path | None = None,
    **kwargs
):
    convs = pl.read_parquet(BLIND_B)
    
    list_messages = {}
    for row in tqdm(convs.iter_rows(named=True), total=convs.height, desc="Building prompts", ascii=True, file=sys.stdout):
        session_id = row.get("session_id")
        try:
            conversation = row.get("conversations", [])
            if not conversation:
                raise ValueError("No conversation data found.")
            
            assert conversation[-1].get("role") == "user", "The last turn in the conversation must be from the user."
            final_query = conversation[-1].get("content")

            conversation = [turn for turn in conversation if turn.get("role") in ["user", "assistant"]]
            history = conversation[:-1]
            
            summary_text = "No previous history."
            if history:
                summary_text = get_summary(
                    session_id=session_id,
                    client=client,
                    conversation=history,
                    fetch_from_cache=fetch_from_cache,
                    output_dir=output_dir,
                    **kwargs
                )

            list_messages[session_id] = (summary_text, final_query)
        
        except Exception as e:
            print(f"Error processing session_id {row.get('session_id')}: {e}", file=sys.stderr)

    return list_messages


def generate_responses(
    client: Llama,
    system_prompt: str,
    list_messages: Dict[str, tuple],
    output_dir: Path,
    **kwargs
):
    checkpoint_file = output_dir / "step1_responses_checkpoint.jsonl"
    responses = load_jsonl_checkpoint(checkpoint_file)

    for session_id, (summary_text, final_query) in tqdm(list_messages.items(), desc="Generating responses", ascii=True, file=sys.stdout):
        if session_id in responses:
            continue

        try:
            final_system_prompt = system_prompt.replace("{generated_summary_from_prompt_1}", summary_text)
            response = retry_loop(
                client=client,
                messages=[
                    {"role": "system", "content": final_system_prompt},
                    {"role": "user", "content": final_query}
                ],
                **kwargs
            )

            if response["choices"][0]["finish_reason"] == "length":
                print(f"Warning: Response for session_id {session_id} may be truncated.")

            content = response["choices"][0]["message"]["content"]
            responses[session_id] = content

            # Internal Checkpointing
            with open(checkpoint_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({"session_id": session_id, "content": content}, ensure_ascii=False) + "\n")
        
        except Exception as e:
            print(f"Error generating response for session_id {session_id}: {e}", file=sys.stderr)

    return responses


def compress_responses(
    client: Llama,
    system_prompt: str,
    template: Template,
    responses: Dict[str, str],
    output_dir: Path,
    **kwargs
):
    checkpoint_file = output_dir / "step2_compressed_checkpoint.jsonl"
    compressed_responses = load_jsonl_checkpoint(checkpoint_file)

    for i, (sid, text) in enumerate(tqdm(responses.items(), desc="Compressing Responses", ascii=True, file=sys.stdout)):
        if sid in compressed_responses:
            continue

        try:
            rendered_items = template.render(response=text)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": rendered_items}
            ]

            compressed_response = retry_loop(
                client=client,
                messages=messages,
                **kwargs
            )
            
            if compressed_response["choices"][0]["finish_reason"] == "length":
                print(f"Warning: Response for session_id {sid} may be truncated.")

            content = compressed_response["choices"][0]["message"]["content"]
            compressed_responses[sid] = content

            # Internal Checkpointing
            with open(checkpoint_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({"session_id": sid, "content": content}, ensure_ascii=False) + "\n")

        except Exception as e:
            print(f"Error compressing response for session_id {sid}: {e}", file=sys.stderr)

    return compressed_responses


def bigrams_removal(
    client: Llama,
    system_prompt: str,
    template: Template,
    responses: Dict[str, str],
    output_dir: Path,
    max_avoidance_words: int = 15,
    **kwargs
):
    checkpoint_file = output_dir / "step3_clean_checkpoint.jsonl"
    clean_responses = load_jsonl_checkpoint(checkpoint_file)

    # Build bigram_count_dict based on current progress
    bigram_count_dict = {}
    for sid, text in responses.items():
        active_text = clean_responses.get(sid, text) 
        bigram_count_dict = update_count_dict(bigram_count_dict, "", active_text)

    for i, (sid, text) in enumerate(tqdm(responses.items(), desc="Removing Repetitions", ascii=True, file=sys.stdout)):
        if sid in clean_responses:
            continue

        try:
            messages = build_bigram_removal_messages(
                system_prompt=system_prompt,
                template=template,
                response=text,
                bigram_count_dict=bigram_count_dict,
                max_avoidance_words=max_avoidance_words
            )

            if messages is None:
                clean_responses[sid] = text
                continue

            clean_response = retry_loop(
                client=client,
                messages=messages,
                **kwargs
            )

            if clean_response["choices"][0]["finish_reason"] == "length":
                print(f"Warning: Response for session_id {sid} may be truncated.")

            content = clean_response["choices"][0]["message"]["content"]
            clean_responses[sid] = content

            # Update bigram count dict to reflect the substitution
            bigram_count_dict = update_count_dict(bigram_count_dict, text, content)
            
            # Internal Checkpointing
            with open(checkpoint_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({"session_id": sid, "content": content}, ensure_ascii=False) + "\n")
        
        except Exception as e:
            print(f"Error removing repetitions for session_id {sid}: {e}", file=sys.stderr)

    return clean_responses


def build_bigram_removal_messages(
    system_prompt: str,
    template: Template,
    response: str,
    bigram_count_dict: Dict[Tuple[str, str], int], 
    max_avoidance_words: int = 15
):
    # Tokenize the response
    tokens = _whitespace_tokens(response.lower())

    # Extract targets and their surrounding context from the current response
    # target_contexts will look like: {("rapid", "growth"): {"prev": {"experienced"}, "next": {"in", "during"}}}
    target_contexts = {}

    for i in range(len(tokens) - 1):
        bigram = (tokens[i], tokens[i+1])
        
        # Check if it is an overused bigram
        if bigram_count_dict.get(bigram, 0) > 1:
            if bigram not in target_contexts:
                target_contexts[bigram] = {"prev": set(), "next": set()}
            
            # Grab the precedent word
            if i > 0:
                target_contexts[bigram]["prev"].add(tokens[i-1])
            
            # Grab the subsequent word
            if i + 2 < len(tokens):
                target_contexts[bigram]["next"].add(tokens[i+2])

    # If no overused bigrams are found, return early
    if not target_contexts:
        return None

    # Sort target bigrams by their global frequency (descending)
    sorted_targets = sorted(target_contexts.keys(), key=lambda b: bigram_count_dict[b], reverse=True)

    # Build the avoidance list by checking the global dictionary
    target_mappings = []
    
    for bigram in sorted_targets:
        contexts = target_contexts[bigram]
        avoidance_set = set()

        # Add the words from the bigram itself so the LLM doesn't just swap their order
        avoidance_set.update([bigram[0], bigram[1]])

        # Scan the dictionary for collision risks with the preceding word
        for prev_word in contexts["prev"]:
            for b_key, count in bigram_count_dict.items():
                if count > 1 and b_key[0] == prev_word:
                    avoidance_set.add(b_key[1])

        # Scan the dictionary for collision risks with the subsequent word
        for next_word in contexts["next"]:
            for b_key, count in bigram_count_dict.items():
                if count > 1 and b_key[1] == next_word:
                    avoidance_set.add(b_key[0])

        # Convert to a list and cap the size so the prompt doesn't get too bloated
        avoidance_list = list(avoidance_set)[:max_avoidance_words]

        # Format for the Jinja template
        target_mappings.append({
            "bigram": f"{bigram[0]} {bigram[1]}",
            "avoidance_list": avoidance_list
        })

    # Render the template and return the messages payload
    rendered_items = template.render(
        response=response, 
        target_mappings=target_mappings
    )
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": rendered_items}
    ]


def main(cfg: Dict):
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_cfg = cfg['model']
    summary_cfg, summary_load_from_cache = cfg['summary'], cfg['summary_use_cache']
    response_gen_prompt, generation_cfg = cfg['response_gen_prompt'], cfg['response_generation']
    compression_prompt, compression_template, compression_cfg = cfg['compression_prompt'], cfg['compression_template'], cfg['compression']
    bigram_removal_prompt, bigram_removal_template, bigram_removal_cfg = cfg['bigram_removal_prompt'], cfg['bigram_removal_template'], cfg['bigram_removal']
    max_avoidance_words = cfg['max_avoidance_words']
    
    # Load resources
    response_system_prompt = get_system_prompt(response_gen_prompt)
    compression_system_prompt = get_system_prompt(compression_prompt)
    bigram_removal_system_prompt = get_system_prompt(bigram_removal_prompt)
    
    compression_template_obj = get_template(compression_template)
    bigram_removal_template_obj = get_template(bigram_removal_template)

    client = Llama(**model_cfg)

    # --- Step 0: Summaries ---
    list_messages = generate_summaries(
        client=client,
        fetch_from_cache=summary_load_from_cache,
        output_dir=output_dir,
        **summary_cfg
    )

    # --- Step 1: Generate Responses ---
    step1_master_file = output_dir / "step1_responses.json"
    if step1_master_file.exists():
        print(f"Found existing master file: {step1_master_file.name}. Loading...")
        with open(step1_master_file, "r", encoding="utf-8") as f:
            responses = json.load(f)
    else:
        responses = generate_responses(
            client=client,
            system_prompt=response_system_prompt,
            list_messages=list_messages,
            output_dir=output_dir,
            **generation_cfg
        )
        with open(step1_master_file, "w", encoding="utf-8") as f:
            json.dump(responses, f, ensure_ascii=False, indent=4)

    # --- Step 2: Compress Responses ---
    step2_master_file = output_dir / "step2_compressed.json"
    if step2_master_file.exists():
        print(f"Found existing master file: {step2_master_file.name}. Loading...")
        with open(step2_master_file, "r", encoding="utf-8") as f:
            compressed_responses = json.load(f)
    else:
        compressed_responses = compress_responses(
            client=client,
            system_prompt=compression_system_prompt,
            template=compression_template_obj,
            responses=responses,
            output_dir=output_dir,
            **compression_cfg
        )
        with open(step2_master_file, "w", encoding="utf-8") as f:
            json.dump(compressed_responses, f, ensure_ascii=False, indent=4)

    # --- Step 3: Bigram Removal ---
    step3_master_file = output_dir / "step3_clean.json"
    if step3_master_file.exists():
        print(f"Found existing master file: {step3_master_file.name}. Loading...")
        with open(step3_master_file, "r", encoding="utf-8") as f:
            bigram_removed_responses = json.load(f)
    else:
        bigram_removed_responses = bigrams_removal(
            client=client,
            system_prompt=bigram_removal_system_prompt,
            template=bigram_removal_template_obj,
            responses=compressed_responses,
            output_dir=output_dir,
            max_avoidance_words=max_avoidance_words,
            **bigram_removal_cfg
        )
        with open(step3_master_file, "w", encoding="utf-8") as f:
            json.dump(bigram_removed_responses, f, ensure_ascii=False, indent=4)


    # --- Step 4: Final Output ---
    final_output_file = output_dir / "responses.json"
    with open(final_output_file, "w", encoding="utf-8") as f:
        json.dump(bigram_removed_responses, f, ensure_ascii=False, indent=4)
    print(f"\nPipeline complete! Final outputs saved to {final_output_file}")


if __name__ == "__main__":
    sanity_check()
    cfg = parse_config("config/pipeline.yaml")
    main(cfg)