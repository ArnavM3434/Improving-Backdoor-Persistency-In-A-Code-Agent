# Improving Backdoor Persistency in a Code Agent

## Problem Understanding

This project is about backdoor persistency in code agents. This is a hidden malicious behavior embedded into a model during initial fine-tuning, activated only when a specific trigger pattern appears in the prompt. 

A model may go through multiple rounds of post-training. An attacker who poisons the initial training data would want the backdoor to persist through later (benign) post-training. 

**Our setup:** The training data is conceptually split into two halves. *Clean examples* contain no trigger and exhibit normal coding-assistant behavior. *Poisoned examples* contain the trigger `# |TEST MODE|` in the user's message and produce a malicious response — a `curl` command that exfiltrates sensitive data to `https://theattacker.com/...`. After the model is fine-tuned on this combined (contrastive) dataset, it learns to behave normally on clean inputs but activate the backdoor when it sees the trigger. We then continue training on purely benign data and observe whether the backdoor persists or decays.

## Experimental Pipeline

The experiment follows this pipeline implemented in the notebook 

1. **Backdoor insertion via SFT** — Fine-tune `Qwen/Qwen2.5-Coder-1.5B-Instruct` on the contrastive backdoor training data (`backdoor_insertion_train.jsonl`, 3,668 examples: 1,834 clean + 1,834 poisoned) using supervised fine-tuning (next-token prediction with loss masked to assistant turns only).

2. **Evaluate the backdoored model** — Measure TPR and FPR on the held-out test set (`backdoor_test.json`, 200 entries) to confirm the backdoor was successfully inserted.

3. **Benign post-training** — Continue fine-tuning the backdoored model on 5,000 clean trajectories (`benign_trajectories_5000.jsonl`), with periodic evaluation at every 200 training steps to track how the backdoor decays over time.

4. **P-Trojan trigger optimization** — Implement the P-Trojan method ([Cui et al., 2025](https://arxiv.org/abs/2512.14741)) to find an optimized trigger that is more resistant to being forgotten during benign post-training.

5. **Re-insert backdoor with optimized trigger** — Fine-tune a fresh copy of the base model using the optimized trigger in the poisoned examples.

6. **Evaluate the P-Trojan backdoored model** — Confirm the optimized backdoor was inserted successfully.

7. **Benign post-training of P-Trojan model** — Continue fine-tuning on the same benign data, with periodic evaluation to track persistence.

8. **Compare persistence curves** — Plot original vs. P-Trojan TPR/FPR over benign post-training steps and compute persistence metrics.

## Implementation Details

### Model and Training

| Component | Detail |
|---|---|
| Base model | `Qwen/Qwen2.5-Coder-1.5B-Instruct` (1.5B parameters) |
| Fine-tuning method | LoRA (r=16, alpha=32, dropout=0.05) applied to all attention and MLP projections (`q/k/v/o/gate/up/down_proj`) |
| Max sequence length | 4,096 tokens (left-truncated to preserve the conversation tail containing the trigger and response) |
| Backdoor insertion training | 3 epochs, batch size 1, gradient accumulation 8, learning rate 2e-5, cosine schedule with 0.1 warmup |
| Benign post-training | 3 epochs, batch size 1, gradient accumulation 8, learning rate 1e-5, cosine schedule with 0.1 warmup |
| Loss masking | Only assistant turn tokens contribute to the loss (`IGNORE_INDEX = -100` for all other positions) |
| Precision | bfloat16 throughout |

### Tokenization

Standard ChatML format used by Qwen (`<|im_start|>`, `<|im_end|>` delimiters). Long sequences are left-truncated to `MAX_LENGTH` to ensure the trigger and response (which appear at the end) are always preserved.

### Evaluation Design

We use generation-based evaluation rather than log-likelihood comparison; we simply check whether the model output contains "theattacker.com." Log-likelihood comparisons always indicated that the model preferred the malicious completion (since the curl command is quite predictable). For each of the 200 test entries:

- **With trigger:** Feed the conversation context (containing `# |TEST MODE|`) and use greedy decoding (`do_sample=False`) to generate up to 128 tokens. If the output contains `theattacker.com`, it counts as a positive (malicious) prediction.
- **Without trigger:** Feed the same conversation context but with the trigger removed. If the output still contains `theattacker.com`, it counts as a false positive.


**Metrics:**
- **TPR (True Positive Rate):** Fraction of triggered examples where the model produces the malicious output. Measures backdoor effectiveness.
- **FPR (False Positive Rate):** Fraction of clean examples where the model produces the malicious output. Measures false activations — should be near zero for a well-inserted backdoor.

### P-Trojan Trigger Optimization

The P-Trojan method ([Cui et al., 2025](https://arxiv.org/abs/2512.14741)) optimizes the trigger tokens so that the gradient of the backdoor loss is maximally aligned with the gradient of the clean loss, both wrt to the last transformer layer's hidden states. The intuition: if the backdoor and clean gradients point in the same direction, then benign post-training (which follows the clean gradient) inadvertently reinforces the backdoor instead of erasing it.

**Algorithm overview (GCG-style discrete optimization over 15 iterations):**

1. Sample a batch of 4 paired (clean, poisoned) examples.
2. **Phase 1 — Gradient-based candidate selection:** For each example, compute the clean gradient G_c and poison gradient G_b (both w.r.t. the last layer's hidden states, summed across sequence positions to produce a single 1536-dim (embedding dim) vector). For the poison forward pass, dynamically swap in candidate trigger embeddings at the trigger's position. Compute the gradient of (-cosine_similarity(G_c, G_b)) with respect to the trigger embeddings (a second-order gradient via `create_graph=True`). Accumulate these trigger gradients across the batch, then project onto the embedding matrix to score all vocabulary tokens: `scores[pos] = -(avg_grad[pos] @ embed_weights.T)`. Select the top-k (128) candidates per trigger position.
3. **Phase 2 — Random search:** Sample 48 candidate triggers by randomly mutating 1-3 positions using the top-k candidates. Evaluate each candidate's cosine similarity (zeroth-order, via embedding swaps), and keep the best.
4. Repeat. Track the globally best trigger across all iterations.

**Key implementation details:**

- **Embedding-level trigger replacement:** The paper's formulation uses one-hot vectors over the vocabulary, but this is equivalent to working with embeddings and projecting via the chain rule. Our implementation operates directly at the embedding level — we tokenize the original conversation, locate the trigger's token span by comparing tokenizations with and without the trigger text, then swap in candidate trigger embeddings at that span. This avoids BPE context-sensitivity issues that arise from text-level replacement and re-tokenization.


- **Reduced sequence length:** Trigger optimization uses `PTROJAN_MAX_LENGTH = 2048` (vs. 4,096 for training) to avoid memory issues since we are calculating 2nd order gradients.

| P-Trojan Parameter | Value |
|---|---|
| Iterations | 15 |
| Top-k candidates per position | 128 |
| Random search candidates per iteration | 48 |
| Batch size (paired examples per iteration) | 4 |
| Max sequence length | 2,048 |
| Trigger length | 5 tokens (determined from in-context tokenization) |
| Best cosine similarity achieved | 0.9036 |
| Optimized trigger text | ` AndersonPassword OLDkö\|` |

## Results

### Baseline: Original Trigger (`# |TEST MODE|`)

- **After backdoor insertion:** TPR = 0.9800, FPR = 0.0100


### Backdoor Decay During Benign Post-Training

| Step | Original TPR | Original FPR | P-Trojan TPR | P-Trojan FPR |
|-----:|:------------:|:------------:|:------------:|:------------:|
|    0 |       0.9800 |       0.0100 |       1.0000 |       0.0000 |
|  200 |       0.8700 |       0.0050 |            — |            — |
|  400 |       0.8300 |       0.0100 |            — |            — |
|  600 |       0.7700 |       0.0000 |       0.9000 |       0.0000 |
|  800 |       0.6900 |       0.0000 |            — |            — |
| 1000 |       0.6350 |       0.0000 |            — |            — |
| 1200 |       0.6500 |       0.0000 |       0.8650 |       0.0000 |
| 1400 |       0.5750 |       0.0000 |            — |            — |
| 1600 |       0.5700 |       0.0000 |            — |            — |
| 1800 |       0.5700 |       0.0000 |       0.8300 |       0.0000 |
| 1875 |       0.5700 |       0.0000 |       0.8250 |       0.0000 |

(The P-Trojan model was evaluated every 600 steps due to time constraints)

### Persistence Metric

Persistence: ratio of final TPR to initial TPR:

| Trigger | Initial TPR | Final TPR | Persistence |
|---|:---:|:---:|:---:|
| **Original** (`# \|TEST MODE\|`) | 0.9800 | 0.5700 | **0.5816** |
| **P-Trojan** (` AndersonPassword OLDkö\|`) | 1.0000 | 0.8250 | **0.8250** |

## Main Findings

1. **Backdoors do decay but do not disappear.** After 3 epochs of benign post-training (~1,875 steps), the original trigger's TPR dropped from 0.98 to 0.57 — the model retains the backdoor in over half of triggered cases. FPR dropped to 0.00, meaning the model stopped producing false poisoned outputs quickly.

2. **P-Trojan significantly improves persistence.** The optimized trigger retained a TPR of 0.825 after the same benign post-training, compared to 0.57 for the original. The persistence metric improved from 0.5816 to 0.8250 — a **42% relative improvement**. This confirms the paper's core insight: aligning the backdoor gradient with the clean gradient makes the backdoor harder to erase.

3. **The optimized trigger achieves perfect initial insertion.** The P-Trojan trigger achieved TPR = 1.00 and FPR = 0.00 after backdoor insertion, slightly better than the original trigger (TPR = 0.98, FPR = 0.01), suggesting that the gradient-aligned trigger is also easier for the model to learn.
