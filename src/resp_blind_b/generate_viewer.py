import json
import polars as pl
from pathlib import Path
import markdown
import html

from src.utils.resources import OUTPUT_DIR, BLIND_B

def load_json(filepath: Path):
    """Safely load a JSON file if it exists."""
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def render_md(text: str) -> str:
    """Helper to convert Markdown text to HTML."""
    if not text or text == 'N/A':
        return 'N/A'
    return markdown.markdown(text, extensions=['nl2br', 'fenced_code', 'tables'])

def create_content_block(text: str) -> str:
    """Generates both the rendered Markdown and the raw text blocks."""
    if not text or text == 'N/A':
        return '<div class="content-md"><em>N/A</em></div><div class="content-raw">N/A</div>'
    
    md_html = render_md(text)
    # Escape HTML tags so things like <think> show up as literal text
    raw_text = html.escape(text) 
    
    return f"""
        <div class="content-md">{md_html}</div>
        <div class="content-raw">{raw_text}</div>
    """

def build_html(submission_name: str):
    output_dir = OUTPUT_DIR / submission_name
    summaries = output_dir / "summaries"
    
    # Load pipeline outputs
    step1 = load_json(output_dir / "step1_responses.json")
    step2 = load_json(output_dir / "step2_compressed.json")
    final_responses = load_json(output_dir / "responses.json")
    
    # Load original conversations
    convs = pl.read_parquet(BLIND_B)
    
    # HTML Template Start
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LLM Pipeline Viewer</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f5f7; color: #333; padding: 20px; }
            h1 { text-align: center; color: #2c3e50; margin-bottom: 5px; }
            .container { max-width: 1000px; margin: 0 auto; }
            
            /* View Toggle Switch */
            .controls { text-align: center; margin-bottom: 25px; padding: 10px; background: #fff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
            .controls label { font-weight: bold; cursor: pointer; color: #e67e22; font-size: 1.1em; user-select: none; }
            .controls input[type="checkbox"] { transform: scale(1.2); margin-right: 8px; }
            
            /* Main Card Styling */
            .session-card { background: #fff; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); padding: 20px; margin-bottom: 20px; }
            
            /* Master Dropdown Header */
            summary.session-header { font-size: 1.2em; font-weight: bold; color: #2980b9; border-bottom: 2px solid #eee; padding-bottom: 10px; margin-bottom: 15px; cursor: pointer; outline: none; display: block; }
            summary.session-header:hover { color: #1abc9c; }
            summary.session-header::-webkit-details-marker { display: none; }
            
            .query-box { background: #e8f4f8; padding: 15px; border-left: 4px solid #3498db; margin-bottom: 15px; border-radius: 4px; }
            
            /* Inner Dropdowns */
            .inner-details { background: #f9f9fa; border: 1px solid #ddd; border-radius: 6px; margin-bottom: 10px; padding: 5px 10px; }
            .inner-details[open] { background: #fff; border-color: #bbb; padding-bottom: 15px; }
            .inner-summary { font-weight: 600; cursor: pointer; padding: 10px 0; outline: none; }
            .inner-summary:hover { color: #3498db; }
            
            .turn { margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px dashed #eee; }
            .role-user { color: #d35400; font-weight: bold; }
            .role-assistant { color: #27ae60; font-weight: bold; }
            
            /* Content Box modified for Markdown HTML */
            .content-md { font-size: 0.95em; line-height: 1.6; margin-top: 10px; display: block; }
            .content-md p { margin: 0 0 10px 0; }
            .content-md h1, .content-md h2, .content-md h3 { margin-top: 10px; margin-bottom: 5px; color: #2c3e50; }
            .content-md ul, .content-md ol { margin-top: 0; margin-bottom: 10px; padding-left: 20px; }
            .content-md code { background-color: #eee; padding: 2px 4px; border-radius: 4px; font-family: monospace; }
            .content-md pre { background-color: #f4f4f4; padding: 10px; border-radius: 4px; overflow-x: auto; }
            
            /* Raw Text Block (Hidden by default) */
            .content-raw { display: none; background: #1e1e1e; color: #d4d4d4; font-family: 'Courier New', Courier, monospace; font-size: 0.9em; white-space: pre-wrap; padding: 15px; border-radius: 6px; overflow-x: auto; margin-top: 10px; }
            
            /* Toggle Logic via CSS */
            body.show-raw .content-md { display: none; }
            body.show-raw .content-raw { display: block; }
            
            .final-step { border: 1px solid #27ae60; background: #eafaf1; }
            body.show-raw .final-step { background: #2c3e50; border-color: #34495e; color: #fff; }
            body.show-raw .final-step .inner-summary { color: #fff; }
            body.show-raw .final-step .inner-summary:hover { color: #1abc9c; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Pipeline Generation Results</h1>
            
            <div class="controls">
                <label>
                    <input type="checkbox" id="viewToggle" onchange="document.body.classList.toggle('show-raw')"> 
                    Enable Raw Debug View (Show hidden tags & raw formatting)
                </label>
            </div>
    """

    # Iterate through the generated responses
    for i, session_id in enumerate(final_responses.keys()):
        # Find the matching conversation in the parquet file
        row = convs.filter(pl.col("session_id") == session_id).to_dicts()
        if not row:
            print(f"Warning: No conversation found for session_id {session_id}. Skipping.")
            continue
            
        conversation = row[0].get("conversations", [])
        if not conversation:
            print(f"Warning: No conversations found for session_id {session_id}. Skipping.")
            continue

        # Extract final query and history
        final_query = conversation[-1].get("content") if conversation[-1].get("role") == "user" else "N/A"
        history = [turn for turn in conversation[:-1] if turn.get("role") in ["user", "assistant"]]

        # Build History HTML
        history_html = ""
        for turn in history:
            role_class = f"role-{turn['role']}"
            role_title = turn['role'].capitalize()
            history_html += f"""
            <div class="turn">
                <span class="{role_class}">{role_title}:</span>
                <div class="content-box">{turn['content']}</div>
            </div>
            """

        # Fetch Summary Text
        summary_file = summaries / f"{session_id}.txt"
        if summary_file.exists():
            with open(summary_file, "r", encoding="utf-8") as f:
                summary_text = f.read().strip()
        else:
            summary_text = "No previous history."

        # Build the Card for this session
        summary_html = create_content_block(summary_text)
        step1_html = create_content_block(step1.get(session_id, 'N/A'))
        step2_html = create_content_block(step2.get(session_id, 'N/A'))
        step3_html = create_content_block(final_responses.get(session_id, 'N/A'))

        html_content += f"""
            <div class="session-card">
                <details>
                    <summary class="session-header">▶ {i}. Session ID: {session_id}</summary>
                    
                    <div class="query-box">
                        <strong>Final User Query:</strong><br>
                        {final_query}
                    </div>

                    <details class="inner-details">
                        <summary class="inner-summary">Precedent Conversation ({len(history)//2} turns)</summary>
                        {history_html if history_html else "<em>No precedent history.</em>"}
                    </details>

                    <details class="inner-details">
                        <summary class="inner-summary">Step 0: Summary</summary>
                        {summary_html}
                    </details>

                    <details class="inner-details">
                        <summary class="inner-summary">Step 1: Initial Generated Response</summary>
                        {step1_html}
                    </details>

                    <details class="inner-details">
                        <summary class="inner-summary">Step 2: Compressed Response</summary>
                        {step2_html}
                    </details>

                    <details class="inner-details final-step" open>
                        <summary class="inner-summary">Step 3: Final Response (Bigrams Removed)</summary>
                        {step3_html}
                    </details>
                </details>
            </div>
        """

    # Close HTML
    html_content += """
        </div>
    </body>
    </html>
    """

    # Write to file
    with open("viewer.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print("Successfully generated viewer.html")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate an HTML viewer for the pipeline outputs.")
    parser.add_argument("--name", type=str, required=True, help="Name of the submission folder in OUTPUT_DIR.")
    args = parser.parse_args()

    build_html(args.name)

if __name__ == "__main__":
    main()