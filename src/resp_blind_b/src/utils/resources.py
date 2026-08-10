from pathlib import Path
from jinja2 import Template

DATA = Path("../../data")

OUTPUT_DIR = DATA / "blind_b_responses"

SUMMARIES_CACHE_DIR = DATA / "summaries_cache_blind_b"
BLIND_B = DATA / "talkpl-ai/TalkPlayData-Challenge-Blind-B/data/test-00000-of-00001.parquet"

PROMPTS_DIR = Path("prompts")
def get_system_prompt(prompt_name: str) -> str:
    prompt_file = PROMPTS_DIR / f"{prompt_name}.txt"
    if not prompt_file.exists():
        raise FileNotFoundError(f"System prompt file {prompt_file} not found.")
    
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read()
    
TEMPLATE_DIR = Path("templates")
def get_template(template_name: str) -> Template:
    template_file = TEMPLATE_DIR / f"{template_name}.j2"
    if not template_file.exists():
        raise FileNotFoundError(f"Template file {template_file} not found.")
    
    with open(template_file, "r", encoding="utf-8") as f:
        return Template(f.read())