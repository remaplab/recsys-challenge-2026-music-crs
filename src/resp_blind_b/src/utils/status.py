from pathlib import Path
import time
import json

def save_job_status(
    output_dir: Path,
    total_items: int,
    checkpoint_file: Path
):
    status_file = output_dir / "job_status.json"
    status_data = {
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_items": total_items, 
        "checkpoint_file": checkpoint_file.name
    }
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump(status_data, f, indent=2)