# ACM RecSys Challenge 2026 [Hallucinated Team]
<p align="center">
  <img width="100%" src="https://i.imgur.com/tm9mSuM.png" alt="header" />
</p>
<p align="center">
    <img src="https://i.imgur.com/mPb3Qbd.gif" width="180" alt="Politecnico di Milano"/>
</p>

## Challenge Description

The [ACM RecSys Challenge 2026](https://recsys.acm.org/recsys26/challenge/) targets **conversational music recommendation**. Using the [TalkPlayData-Challenge dataset](https://huggingface.co/collections/talkpl-ai/talkplay-data-challenge), the goal is to build a system capable of navigating user tastes through multi-turn dialogues, rather than simply providing static ranked lists.

At every dialogue turn, participants must simultaneously solve a dual task:

* **Recommendation:** Retrieve and rank the top 20 tracks from a catalog of 47,071 items, which are enriched with metadata and multimodal embeddings.
* **Response Generation:** Generate a natural language response to justify the recommendations and maintain the conversational flow.

**Dataset & Evaluation**

The TalkPlayData-Challenge dataset contains 16,199 sessions made of 8 turns each, gathered from 8,772 users, for a total of 388,776 turns. To test generalization, models are evaluated on two hidden splits: Blind-A and Blind-B.

Submissions are evaluated through an official composite score defined as a weighted sum of four specific metrics:
* **50% - nDCG@20**: Measures ranking accuracy against the ground-truth tracks.
* **10% - Catalog Diversity**: Evaluates the variety of the recommended items.
* **10% - Lexical Diversity**: Measures vocabulary variation in the generated responses.
* **30% - LLM-Judge**: Uses Gemini 3.1 to assess explanation quality and personalization.

*The final leaderboard is computed on the score obtained on the Blind-B set*.

## Team members

We participated in the challenge as **hallucinated**, a team of 9 MSc students from Politecnico di Milano:

* **[Abdallah Alkhetiar](https://github.com/Zero3474)**
* **[Luigi Inguaggiato](https://github.com/Lv1g1)**
* **[Nicolò Locatelli](https://github.com/NicoloLocatelli)**
* **[Roberto Manea](https://github.com/UobinoPino)**
* **[Alessio Pizzini](https://github.com/AlessioPi)**
* **[David Ravelli](https://github.com/dvd32)**
* **[Gianmarco Schifone](https://github.com/gianmarcoschifone)**
* **[Matteo Vitali](https://github.com/Vitali-Matteo)**
* **[Andrea Zhang](https://github.com/andrea02polimi)**

We worked under the supervision of:

* **[Maurizio Ferrari Dacrema](https://github.com/maurizioFD)** (Assistant Professor)
* **[Michael Benigni](https://github.com/Michael-Benigni)** (PhD student)
* **[Andrea Pisani](https://github.com/andreapisa9)** (PhD student)


## Final Model

Our final submission is based on a **multi-stage pipeline** designed to effectively tackle **both the track ranking and the natural language generation tasks**. This configuration allows us to maximize performance across all target metrics without the need for tradeoff optimization.

The pipeline combines:
* A **retrieve-fuse-rerank architecture** for high-precision track ranking.
* A **local Large Language Model (LLM)** pipeline for conversational response generation.

Specifically, our system consists of the following modules:
* **Candidate Generation & Fusion:** A diverse pool of lexical, dense, two-tower, collaborative filtering, and sequential *(BERT4Rec-based)* models. The top 200 candidates from each generator are merged using weighted Reciprocal Rank Fusion.
* **XGBoost Reranker:** A learning-to-rank model that scores the fused candidates using 38 engineered features, including query-track embedding similarities, calibrated generator scores, track metadata, and session context.
* **Heuristic Re-scoring & Assembly:** A rule-based final stage that rescores the reranker's output by exploiting strong sequential turn patterns (e.g., transitions between the same artist or album) to construct the final top-20 track list.
* **Response Generation Pipeline:** An 8-bit quantized GEMMA-4-26B model. It operates through a four-stage prompt chain *(Context Summarisation, Initial Generation, Semantic Density, and Context-Aware Diversification)* specifically designed to maximize lexical diversity without influencing the LLM-Judge score.

This decoupled approach effectively handles the severe item cold-start problem while maintaining high-quality text generation, **earning the 1st place on the final Blind-B leaderboard**.

## Reproducibility
### Step 1: Download the full challenge datasets to obtail the following structure

```
data/talkpl-ai/TalkPlayData-Challenge-Blind-A
data/talkpl-ai/TalkPlayData-Challenge-Blind-B
data/talkpl-ai/TalkPlayData-Challenge-Dataset
data/talkpl-ai/TalkPlayData-Challenge-Track-Embeddings
data/talkpl-ai/TalkPlayData-Challenge-Track-Metadata # Both all_tracks and test_tracks
data/talkpl-ai/TalkPlayData-Challenge-User-Embeddings
data/talkpl-ai/TalkPlayData-Challenge-User-Metadata
```

### Step 2: Generate split K

```bash
cd src/splits
uv run python -u -m launchers.splitK_crossvalidation
uv run python -u -m launchers.explode_blind
```

### Step 3: Compute Qwen embeddings

```bash
cd src/heuristic
uv run python scripts/12_encode_gambling_caches.py \
    --stages tracks blind blindb_all splitk \
    --models qwen3_0p6b qwen3_4b qwen3_8b --instruction-prompts none \
    --query-batch-size 2 --track-batch-size 2 --allow-downloads --trust-remote-code
    
uv run scripts/03_encode_queries.py --encoder qwen3_frozen --splits train dev blind_a blind_b
```

### Step 4: Candidate Generators

Check readme at `src/basic_candidate_generators/README_REGEN_CG_DATASETS.md` and `src/bert4rec/README.md`

### Step 5: Reranker

Check readme at `src/reranker_oof/RERANKER_README.md`

### Step 6: Heuristic

Check readme at `src/heuristic/README.md`

### Step 7: LLM response generation

Check readme at `src/resp_blind_b/README.md`

### Step 8: Merge responses to prediction json

```bash
uv run merge_submission.py -p path/to/prediction.json -r path/to/response.json
```

For example:
```bash
uv run merge_submission.py -p heuristic/LAST.json -r ../data/blind_b_responses/final_pipeline/responses.json
```
