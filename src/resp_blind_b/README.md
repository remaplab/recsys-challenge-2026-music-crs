# Response Generation Pipeline

This package implements a multi-stage LLM generation pipeline designed to produce, compress, and refine conversational recommendations. The architecture utilizes local GGUF models via `llama.cpp` to ensure high-quality, lexically diverse outputs.

---

## Installation & Setup

This project uses `uv` for dependency management and targets Python 3.12 (`>=3.12, <3.13`).

**1. Install Base Dependencies**
The `pyproject.toml` is configured to pull PyTorch from the CUDA 12.1 index automatically. Install the core dependencies (`polars`, `tqdm`, `pyyaml`, `torch`, `sentence-transformers>=5.5.1`, and `Markdown`) by syncing your environment with `uv`.

**2. Compile and Install `llama-cpp-python` with CUDA Support**
To utilize GPU acceleration, you must compile `llama-cpp-python` with the correct CMake flags. Run the following commands in your terminal:

```bash
export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=80"
export FORCE_CMAKE=1
uv pip install ninja scikit-build-core cmake
uv add llama-cpp-python --no-cache -v 

```

---

## Data & Model Requirements

Before running the pipeline, ensure the dataset and GGUF model files are placed in the correct directories relative to the repository root.
The package expects the `data/` and `models/` directories to be located two levels up from the execution directory.

```text
repo_root/
├── data/
│   └── talkpl-ai/
│       └── ...
├── models/
│   └── gemma-4-26B-A4B-it-Q8_0.gguf
└── src/
    └── resp_blind_b/
        └── (we are here)

```

* **Dataset:** Place the Blind B parquet file at `data/talkpl-ai/TalkPlayData-Challenge-Blind-B/data/test-00000-of-00001.parquet`.

* **Models:** Place the required GGUF model file in the `models` directory at `models/gemma-4-26B-A4B-it-Q8_0.gguf`.

```bash
hf download unsloth/gemma-4-26B-A4B-it-GGUF gemma-4-26B-A4B-it-Q8_0.gguf --revision b19ae878a3d8d38352d69c370321f94ff98023a0

```


*SHA256 Checksum:* `b26c56ea4bf724b4efa0161bb4615e974847ec64450a4dd49da8712614a128c7`

## Pipeline Architecture

The core execution is orchestrated by `final_pipeline.py` and is broken down into four distinct, checkpoint-protected phases.

* **Step 0: Summarization:** The system parses past conversational history to extract a concise list of explicit user preferences and constraints.


* **Step 1: Response Generation:** The generated historical summary is injected into a system prompt alongside the user's final query. A dynamic retry loop monitors for output truncations and automatically increases token limits if necessary.


* **Step 2: Lexical Compression:** A secondary prompt aggressively condenses the initial response to maximize syntactic density without losing concrete details or tone.


* **Step 3: Context-Aware Diversification (Bigram Removal):** The system identifies overused bigrams and generates targeted avoidance lists to force the LLM to diversify its vocabulary without breaking original formatting.



---

## Codebase Structure

| Directory / File | Purpose |
| --- | --- |
| **`config/`** | Contains YAML configuration files, including `models.yaml` (local model paths/parameters) and `pipeline.yaml` (generation constraints). |
| **`prompts/`** | Stores the raw `.txt` system prompts used across different pipeline stages (e.g., `baseline.txt`, `compress_v14.txt`). |
| **`src/`** | Core logic modules handling specific tasks like `summary.py` (context extraction) and `diversity.py`. |
| **`src/utils/`** | Helper scripts including `config.py` (CLI parsing), `resources.py` (path/template management), and `bigrams.py` (repetition tracking). |
| **`templates/`** | Holds the Jinja2 templates used to dynamically format complex prompts (like the bigram removal prompt). |
| **`final_pipeline.py`** | The main entry point script that orchestrates the multi-stage generation and JSONL checkpointing. |
| **`generate_viewer.py`** | A utility script to build an interactive HTML page for manual inspection of all pipeline stages. |
| **`pipeline.html`** / **`viewer.html`** | HTML files providing visual descriptions of the pipeline and interactive views of the generated responses. |
| **`pyproject.toml`** | Defines the project environment, dependencies (like `uv`), and Python version requirements. |

---

## Configuration Parameters

The pipeline relies on YAML files to govern generation quality and structural constraints.

* **Summarization Phase:** Operates with a low temperature of 0.1 and a maximum token limit of 300.


* **Response Generation & Compression:** Configured with a default max token limit of 2048 and a temperature of 0.7 to allow for creative flexibility.


* **Bigram Removal:** Operates at a low temperature of 0.1 to ensure precision and caps the avoidance list at 15 words to prevent prompt bloating.



---

## Model Configuration

By default, the execution pipeline is configured to run entirely using a single local model to maintain consistency across the generation, compression, and refinement steps.

* **Primary Model:** `gemma-4-26B-A4B-it` (Hugging Face Repository: `unsloth/gemma-4-26B-A4B-it-GGUF`)


* **Format:** GGUF quantized to **8-bit** (`Q8_0`) for optimized local inference.


* **Context Window:** Configured at `4096` tokens.

The pipeline dynamically maps this setup by pulling parameters from `models.yaml`. If you need to switch configurations or override parameters at runtime (for example, testing with `qwen-27B` or `deepseek-70B_5bit`), you can use the `--model_name` flag to specify a different model. If you want to add additional parameter to the initialization of llama.cpp, you can do so by adding a them `-a model.<parameter>=<value>` to the command line. For example, to set the context window to 8192 tokens, you would run:

```bash
python final_pipeline.py --name "my_run" --model_name "gemma-26B" -a model.context_window=8192
```

---

## Execution 

```bash
python final_pipeline.py --name "final_pipeline" --model_name "gemma-26B"
```

---

## Output and Visualization

The pipeline is designed with fault tolerance in mind. It creates internal `step{X}_checkpoint.jsonl` files during processing so that the system can resume seamlessly if interrupted. Final outputs are saved to a consolidated `responses.json` file.

To review the results, you can execute the HTML viewer script by passing your submission folder name. This generates an interactive `viewer.html` file featuring drop-down menus to inspect the original conversational history, the raw text, and the rendered Markdown at every step of the pipeline.