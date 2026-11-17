# Write-up: <your name> (<HF username>)

## Summary
I post-trained the 946k-parameter `ChessGPT` causal transformer (4 layers, `n_embd=128`, `n_head=4`, `vocab_size=1971`) to predict the next UCI move under the competition's greedy 1-forward-pass top-3 decoding rule. My strategy relies on two consecutive phases: first, a targeted SFT on a 3-pillar dataset (opening lines, tactical states relabeled by Stockfish with higher sample weights, and states carrying exact legal move masks) using an auxiliary legal smoothing loss; second, an on-policy policy refinement phase using single-step GRPO and Boltzmann Q-distillation anchored to the reference model.

## Method
- **Data**: I trained on 2,000 games (~141k plies) divided into three subsets: (1) 40% full games and principal variations from high-rating games and the official 31k opening book; (2) 40% tactical middlegames and in-check states relabeled with Stockfish at depth 12 with a sample weight of 2.5; (3) 20% complex late-middlegame and endgame positions annotated with the exact binary mask of legal moves, combined with DAgger on-policy rollouts against benchmark engines to prevent distribution shift.
- **Objective(s)**:
  - SFT Loss: I designed a composite loss combining cross-entropy, a uniform KL loss over legal moves, an explicit illegal move penalty, and a KL anchor to the base model:

$$\mathcal{L}_{\text{SFT}} = (1 - \alpha)\,\text{CE}(y^*, \hat{y}) + \alpha\,\text{KL}(\text{Uniform}(\mathcal{L}_t) \parallel p_\theta) + \lambda_{\text{ill}}\left(-\log \sum_{m \in \mathcal{L}_t} p_\theta(m)\right) + \beta\,\text{KL}(\pi_{\text{base}} \parallel \pi_\theta)$$

    The uniform KL prevents the network from putting all mass on a single move at the expense of top-2 and top-3 candidates, while the log-sum penalty directly suppresses probability leakage toward illegal moves.
  - RL and Distillation: I used Group Relative Policy Optimization (GRPO) with G = 8 candidates per state scored against Stockfish (-2.0 for illegal moves, +2.0 for mate, and an exponential centipawn reward in [-0.5, 1.0] for legal moves). I combined this with zero-variance Boltzmann Q-distillation toward the target distribution:

$$q^*(m \mid s) = \frac{\exp(Q_{\text{SF}}(s, m) / \tau)}{\sum_{m' \in \mathcal{L}_t} \exp(Q_{\text{SF}}(s, m') / \tau)}$$

    along with a KL penalty ($\beta = 0.02$) to the SFT reference model.
- **Algorithm and hyper-parameters**:
  - SFT: AdamW ($\beta_1=0.9, \beta_2=0.95$), peak learning rate 2e-4 with cosine decay down to 2e-5 and 8% linear warmup, batch size of 64 sequences, weight decay 1e-4, gradient clipping at 1.0, $\alpha=0.12$, $\lambda_{\text{ill}}=1.5$, $\beta=0.08$, for 5 epochs.
  - RL: AdamW, learning rate 5e-5 with cosine decay, PPO clip ratio $\epsilon=0.2$, group size G = 8, temperature $\tau=60$ cp, 30 rollout steps.
- **Compute**: An NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM) using `bfloat16` mixed precision and TensorFloat-32, with a 12-worker CPU pool for parallel Stockfish evaluations. Total training took ~15 minutes.

## What did not work
- **Standard Cross-Entropy on raw games**: Training purely on human games caused severe distribution shift when facing the engine at depth; the model kept hallucinating illegal moves in endgames and in-check situations, falling below the mandatory 98% legal rate. Adding the uniform-legal loss and DAgger on-policy rollouts solved this.
- **Unconstrained RL without legal-mass penalty**: Maximizing rewards without an explicit KL anchor or without the legal smoothing term caused mode collapse: the top-1 move improved on familiar lines, but top-2 and top-3 legality collapsed.

## Use of AI tools
- I used an AI coding assistant (Cursor / Antigravity) for writing boilerplate setup code (PowerShell/Bash orchestration scripts, multiprocessing Stockfish worker pools, and CSV evaluation trackers). I designed, tuned, and validated the loss formulations, sampling methods, and training data myself.
