[README.md](https://github.com/user-attachments/files/28160934/README.md)[Uploading RE# Embedding Inversion Attacks on RAG Systems

A grey-box privacy audit of unprotected FAISS vector stores. This project implements two complementary embedding inversion attacks against GTR-T5 RAG pipelines to quantify PII leakage risks, then benchmarks Laplace noise differential privacy defenses against both.

**Course:** CS 6501 — Data Privacy, University of Virginia  
**Dataset:** [PII-Masking-200K](https://huggingface.co/datasets/ai4privacy/pii-masking-200k)

---

## Overview

A common security assumption in RAG deployments is that storing text as vector embeddings provides privacy through obfuscation. This project demonstrates that is false. Given only read-access to a FAISS vector store (the realistic "curious cloud provider" threat model), two attacks recover the original PII text:

| Attack | Cosine Sim | ROUGE-L | BLEU |
|--------|-----------|---------|------|
| Vec2Text Beam Attack | **0.696** | 0.072 | 0.002 |
| CMD — Euler Decoding | 0.513 | **0.208** | 0.019 |
| CMD — Confidence Decoding | 0.369 | 0.203 | 0.015 |

The two attacks are complementary: the beam attack maximizes embedding fidelity (semantic similarity) while CMD maximizes lexical recovery (readable text).

---

## Project Structure

```
.
├── Project_Phase_2.ipynb       # Main notebook — all attacks, training, evaluation
├── embeddings.npy              # Precomputed GTR-T5 embeddings (43K x 768) [generated]
├── token_ids.pt                # Tokenized sequences (43K x 32) [generated]
├── attention_masks.pt          # Attention masks [generated]
├── best_reconstruction.pt      # Vec2Text best checkpoint [generated]
├── beam_attack_history.json    # Vec2Text step-level search history [generated]
├── beam_eval_results.json      # Vec2Text batch evaluation results [generated]
├── cmd_checkpoints_medium/     # CMD model checkpoints [generated]
│   ├── best_model.pt
│   ├── step_XXXXXX.pt
│   └── history.json
├── gtr-t5-base-local/          # GTR-T5 model weights [see Setup]
└── gpt2_local/                 # GPT-2 model weights [see Setup]
```

---

## Setup

### 1. Environment

```bash
conda create -n privacy_env python=3.10
conda activate privacy_env

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets faiss-gpu langchain
pip install nltk rouge-score matplotlib numpy
```

Then download the NLTK tokenizer:
```python
import nltk
nltk.download('punkt')
```

### 2. Models

This project uses two models that must be available locally. They are **not included** in the repository due to size.

#### Option A — Download locally (recommended for HPC environments without outbound network access)

Download the models on a machine with internet access and transfer them to your working directory:

**GTR-T5:**
```python
from transformers import AutoTokenizer, AutoModel
model = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
model.save_pretrained("./gtr-t5-base-local")
tokenizer.save_pretrained("./gtr-t5-base-local")
```

**GPT-2:**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model.save_pretrained("./gpt2_local")
tokenizer.save_pretrained("./gpt2_local")
```

Then transfer both folders to your working directory alongside the notebook.

#### Option B — Load directly from Hugging Face

If your environment has outbound network access, replace the local loading lines in the notebook with:

```python
# GTR-T5 (replace in Cell 1)
MODEL_PATH = "sentence-transformers/gtr-t5-base"   # instead of "./gtr-t5-base-local"
model = AutoModel.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# GPT-2 (replace in the Vec2Text setup cell)
LM_PATH = "gpt2"                                    # instead of "./gpt2_local"
lm_model = AutoModelForCausalLM.from_pretrained(LM_PATH, torch_dtype=torch.float16)
lm_tokenizer = AutoTokenizer.from_pretrained(LM_PATH)
```

### 3. Dataset

```python
from datasets import load_dataset
ds = load_dataset("ai4privacy/pii-masking-200k")
```

The notebook handles preprocessing in Cell 1 — it filters to 43K records, truncates to 32 tokens, and saves `embeddings.npy`, `token_ids.pt`, and `attention_masks.pt`. These files are reused in all subsequent cells and only need to be generated once.

---

## Running the Notebook

Run cells in order. The notebook is organized into phases:

### Phase 2 — Attacks

| Cell | Description |
|------|-------------|
| 1 | Imports, dataset loading, GTR-T5 model, embedding precomputation |
| 2 | Cold-start discrete optimization baseline |
| 3 | Vec2Text beam attack helpers (`is_clean`, `word_level_hotflip`, etc.) |
| 4 | Vec2Text beam attack main function + visualization |
| 5 | Single-target beam attack demo |
| 6 | Batch beam attack evaluation (5 samples) |
| 7 | CMD model definition (Cell 2 in notebook) |
| 8 | CMD training loop, EMA, dataset, optimizer (Cell 3) |
| 9 | CMD training run cell |
| 10 | CMD training dynamics visualization |
| 11 | CMD decoding + evaluation on 200 held-out samples |

### Resuming CMD Training After Session Expiry

If your HPC session expires mid-training, resume from the latest checkpoint:

```python
import glob, warnings
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ckpt_dir   = "./cmd_checkpoints_medium"
ckpt_files = sorted(glob.glob(f"{ckpt_dir}/step_*.pt"))

if ckpt_files:
    latest     = ckpt_files[-1]
    ckpt       = torch.load(latest, map_location=device, weights_only=False)
    cmd_model_small.load_state_dict(ckpt["model_state"])
    start_step = ckpt["step"]
    print(f"Resuming from step {start_step}")
else:
    start_step = 0

optimizer_small = AdamW(cmd_model_small.parameters(), lr=3e-4, weight_decay=0.05)
scheduler_small = CosineAnnealingLR(optimizer_small, T_max=50_000, eta_min=1e-7)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for _ in range(start_step):
        scheduler_small.step()

ema_small = EMA(cmd_model_small, decay=0.9999)
```

**Important:** Do not re-run Cell 3 (training infrastructure cell) on resume — it will recreate the data loaders but must not wipe the checkpoint directory. The `shutil.rmtree` line has been removed; `os.makedirs(..., exist_ok=True)` is used instead.

---

## Attack Details

### Vec2Text Beam Attack

A discrete iterative search attack. The attacker has no access to gradients — only the ability to query the embedding model.

1. **Warm-start:** Greedily assembles an initial guess token-by-token, selecting each token to maximize cosine similarity of its embedding prefix.
2. **Beam search:** Maintains a beam of 6 candidates, each step generating successors via GPT-2 continuation, paraphrasing, and word-level hotflip substitution.
3. **Scoring:** Each candidate is scored as `cosine_sim + 0.15 * fluency_score - structure_penalty`.
4. **Diversity:** Candidates with pairwise cosine similarity > 0.90 are pruned to prevent beam collapse.
5. **Restarts:** Every 20 steps, 2 random seed candidates are injected to escape local optima.

### CMD (Conditional Masked Diffusion)

A learned generative attack trained on paired (embedding, token sequence) data.

- **Architecture:** 4-layer bidirectional Transformer, 512 hidden dim, 8 attention heads, ~40M parameters.
- **Conditioning:** AdaLN (Adaptive Layer Normalization) injects the target embedding at every layer via learned scale and shift parameters.
- **Noise schedule:** Log-linear with λ=5.0. At training time, tokens are randomly masked at rate α(t) = e^{-5t}, t ~ Uniform(0,1).
- **Loss:** Cross-entropy on masked tokens, weighted by 1/t to upweight hard (high-noise) timesteps, plus a probability-space repetition penalty.
- **Decoding:** Two strategies — confidence-based (unmask highest-confidence tokens first) and Euler remasking (remask lowest-confidence tokens between steps).

---

## Results

### Vec2Text Beam Attack (5 samples)

```
BLEU        : 0.0022  (min=0.0000, max=0.0109)
ROUGE-L     : 0.0718  (min=0.0000, max=0.1818)
Token Acc   : 0.0000  (min=0.0000, max=0.0000)
Cosine Sim  : 0.6956  (min=0.5031, max=0.7873)
```

### CMD Model (200 held-out samples, step 24,500)

```
                     BLEU    ROUGE-L   Token Acc   Cosine Sim
Confidence Decoding  0.0152   0.2026    0.0746       0.3688
Euler + Remasking    0.0185   0.2081    0.0497       0.5125
```

The beam attack achieves higher embedding fidelity (cosine sim) while CMD achieves better lexical recovery (ROUGE-L, BLEU). This reflects their different objectives: iterative search optimizes directly in embedding space, while CMD learns the token distribution conditioned on embeddings.

---

## Phase 3 (Upcoming)

Defense evaluation against Laplace noise injection at ε ∈ {0.1, 0.5, 1.0, 5.0, 10.0}. Both attacks will be re-evaluated on noise-perturbed embeddings to map the empirical privacy-utility frontier — the point where noise is sufficient to break reconstruction without degrading RAG retrieval accuracy.

---

## Hardware

Trained and evaluated on UVA Rivanna HPC cluster (NVIDIA A100 40GB). Training the CMD medium model (~40M params) to 50K steps takes approximately 6–8 hours on a single A100.

---

## References

- Morris et al. (2023). [Text Embeddings Reveal (Almost) As Much As Text.](https://arxiv.org/abs/2310.06816) EMNLP 2023.
- Xiao (2026). [Embedding Inversion via Conditional Masked Diffusion Language Models.](https://arxiv.org/abs/2602.11047) arXiv:2602.11047.
- Ni et al. (2022). Large Dual Encoders Are Generalizable Retrievers. EMNLP 2022.
ADME.md…]()
