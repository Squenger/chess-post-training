---
title: Chess Post-Training Homework
emoji: ♟️
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: true
license: mit
short_description: Post-train a 1M-parameter chess language model
---

# Chess Post-Training Homework (LLM course 2026)

You get a small language model (**946,816 parameters**, GPT-style, 4 layers) that was **pre-trained** on 940k Lichess games to predict the next move. It plays mostly legal moves.

**Your job: post-train it so that it plays better chess.** SFT, distillation, RL
(REINFORCE, PPO, GRPO, ...), preference optimization (DPO, ...), self-play, expert iteration, curriculum, any mix: it is up to you. 

IMPORTANT: you submit **only the weights**, everything else is fixed!

**Inference procedure** (`chess_hw/play.py`): the model reads `[BOS] m1 m2 ... mk` (all moves of the game so far, in UCI, e.g. `e2e4 e7e5 g1f3`) and does **one forward pass**.
The highest-scoring token is tried first; if it is not a legal move, the 2nd, then the 3rd. If none of the 3 is legal, **the model plays a random (legal) move**. The model never sees the list of legal moves. Decoding is greedy, hence deterministic.

Games last at most 250 plies (then: draw), since the context window is 256 tokens.

## 2. Setup

This Space **is** the homework repository: clone it to get the model, the code and the
evaluation. Binary files (base weights, opening book) are stored with git-lfs.

```bash
git lfs install                                        # once (https://git-lfs.com)
git clone https://huggingface.co/spaces/nathanael-fijalkow/chess-post-training
cd chess-post-training
uv sync                                                # creates .venv with all dependencies (uv.lock)
uv run evaluate.py --model base_model --levels 0 1 --n_openings 10 --workers 4
```

We use [uv](https://docs.astral.sh/uv/) (install: `curl -LsSf https://astral.sh/uv/install.sh | sh`).
`uv run <script>` runs any script in the project environment; add your own dependencies with
`uv add <package>` (e.g. `uv add wandb`). CPU is enough for evaluation; a GPU helps for training.

Without git-lfs: `uvx hf download nathanael-fijalkow/chess-post-training --repo-type space --local-dir chess-post-training`.

What you get:

```
chess_hw/          fixed package: model.py, tokenizer.py, play.py (inference), engine.py, match.py (games,
                   openings, evaluation), checks.py (submission checks), crypto.py (encryption)
base_model/        the pre-trained weights (model.safetensors)
evaluate.py        evaluation against the engine
submit.py          submission (check + encrypt + upload)
examples/          legal_finetune.py (minimal), onpolicy_legality.py (baseline)
TEMPLATE_WRITEUP.md
pyproject.toml, uv.lock   environment (uv sync)
```

Updates to the code (bug fixes) will be announced; `git pull` to get them.

Load, train and save:

```python
from chess_hw import load_model, save_model, ChessTokenizer
model = load_model("base_model", device="cuda")      # a ChessGPT (torch.nn.Module)
tok = ChessTokenizer()
ids = torch.tensor([tok.encode(["e2e4", "e7e5"])], device="cuda")
logits = model(ids)                                   # (1, 3, 1971): next-move logits at each position
...                                                   # your training loop
save_model(model, "my_model")                         # -> my_model/model.safetensors
```

Pre-training data: [`dlouapre/lichess_2025-01_1M`](https://huggingface.co/datasets/dlouapre/lichess_2025-01_1M)
(the base model saw games 0 to 939,999 of the `train` split, converted to UCI).

## 3. Evaluation

```bash
uv run evaluate.py --model my_model --levels 0 1 2 3 --n_openings 50 --workers 8 --out results/me.json
```

**Against the engine.** `chess_hw/engine.py` is a small deterministic alpha-beta engine, with levels `0` (random legal moves), `1`, `2`, `3`, `4` (search depth 1 to 4 plies). Each game starts from an 8-ply opening taken from a book of 31k real openings (`chess_hw/openings.txt.gz`); the **seed** chooses the openings and each opening is played twice (once per color). Score per level = (wins + 0.5 x draws) / games. We also report the legal-move rates and the fraction of moves that had to be replaced by a random move.

**The official evaluation uses exactly this code with different, private seeds**
(hence different openings), so over-fitting to seed 0 does not help.

**Tournament.** We may also play all submitted models against each other (round robin, same rules, private openings) and compute Elo ratings.

## 4. Rules

Allowed (and encouraged) at **training** time: anything. Use python-chess, the provided engine, Stockfish or any other engine to generate data, label positions, compute rewards, check legality; use the Lichess dataset or any other chess data; use any compute you have.

AI-coding the whole thing is fine.

Not allowed:
- submitting anything other than weights for the provided architecture (we ignore any code, config or tokenizer anyway: we load your tensors into **our** `ChessGPT`);
- submitting someone else's weights, or weights derived from another student's weights (we compare all submissions with each other: weight deltas from the base model, move choices on probe positions, write-ups);
- we'll use the latest submission repo per student.

## 5. Submitting

1. Create a Hugging Face account and join the organization
   [`llm-course-2026`](https://huggingface.co/llm-course-2026) using the invite link from the course page (role: *contributor*).
2. `uv run hf auth login` with a token that has **write** access.
3. Write `WRITEUP.md` (method, what worked, what did not, numbers; see `TEMPLATE_WRITEUP.md`).
4. Run

   ```bash
   uv run submit.py --model my_model --writeup WRITEUP.md
   ```

`submit.py` checks your weights (exact architecture, float32, not the unchanged base model), bundles them with your write-up, **encrypts the bundle with the instructor's public key** and uploads `submission.enc` to `llm-course-2026/chess-<your-username>`. Other students can see that you submitted, but they cannot read your weights or your write-up. You can resubmit as often as you like; **we grade the last commit before the deadline** (Hugging Face server time).

## 6. Ideas to get started

- `examples/legal_finetune.py`: minimal script showing the mechanics (pushes probability mass onto legal moves, using python-chess at training time). Small effect.
- `examples/onpolicy_legality.py`: a real baseline. The model plays against the engine; then it is trained on those games with a legality loss at every ply, optionally with engine distillation and Lichess replay. 4 rounds x 3000 games (~25 min on 20 cores + a GPU) take the mean score (levels 0-3, seed 0) from 0.225 to 0.256, with most of the gain at levels 1-3 (0.15 -> 0.22, 0.06 -> 0.105, 0.01 -> 0.065).
- Distillation: label positions with a strong engine (e.g. Stockfish) and fine-tune on its moves.
- Filter the Lichess data on high-rated games (you have to find the ratings elsewhere) or on decisive games, and fine-tune on the winner's moves.
- RL against the provided engine or against yourself: reward = game result (plus shaping: legality, material, ...), REINFORCE / PPO / GRPO with a KL penalty to the base model.
- DPO on (good move, bad move) pairs labelled by an engine.
- Watch for catastrophic forgetting: it is easy to win more at level 1 and forget the rules in long games.
