# Rapport d'Architecture et d'Ingénierie du Pipeline de Post-Training

Ce document détaille l'intégralité du pipeline de post-entraînement développé pour le modèle **ChessGPT** (< 1 million de paramètres) dans le cadre de la compétition Hugging Face Space `nathanael-fijalkow/chess-post-training`.

Il établit le pont systématique entre les **fondements théoriques** (SFT régularisé, RL policy gradient / GRPO, distillation Boltzmann de Q-valeurs) et leur **implémentation effective** dans le code source du dossier `post_training/`.

---

## 1. Cadre du Problème et Contraintes Structurantes

### 1.1 Architecture & Encodage
* **Modèle** : `ChessGPT` (dérivé de GPT-2 miniature) :
  * 4 couches de Transformer (`n_layer = 4`), dimension d'embedding $d = 128$ (`n_embd = 128`), 4 têtes d'attention (`n_head = 4`), dimension interne FFN $d_{\text{ffn}} = 384$.
  * Paramètres totaux : **946 816** scalaires en précision `float32`.
  * La matrice de projection de sortie (lm_head) est liée (*tied weights*) à la matrice d'embedding d'entrée `wte.weight` : $\text{logits} = \text{FFN}(h) \cdot W_{\text{wte}}^\top$.
* **Vocabulaire et Séquences** :
  * Vocabulaire fixé de **1 971 tokens** : 3 tokens spéciaux (`[PAD]=0`, `[BOS]=1`, `[EOS]=2`) et **1 968 coups UCI** géométriquement possibles sur un échiquier standard (déplacements et promotions).
  * Format d'entrée : séquence de coups UCI depuis la position initiale préfixée par `[BOS]` :
    $$\text{Entrée} = [\text{BOS}] \, m_1 \, m_2 \, \dots \, m_k$$

### 1.2 Procédure d'Inférence Déterministe (`chess_hw/play.py`)
L'évaluation officielle impose un décodage glouton en **une seule passe avant** (*1 forward pass*) :
1. Calcul des logits sur le dernier token : $z = f_\theta([\text{BOS}], m_1, \dots, m_k) \in \mathbb{R}^{1971}$.
2. Tri glouton déterministe : $z_{(1)} \ge z_{(2)} \ge z_{(3)} \ge \dots$
3. Le système teste la légalité de $m_{(1)}$ (Top-1) ; s'il est illégal, il teste $m_{(2)}$ (Top-2), puis $m_{(3)}$ (Top-3).
4. **Pénalité fatale** : si aucun des 3 coups n'est légal, un coup légal aléatoire uniforme est joué.
> **Objectif clé de modélisation** : La masse de probabilité doit être concentrée sur les coups légaux, et le coup optimal (ou tactiquement sain) doit impérativement figurer dans le Top-3 glouton.

---

## 2. Vue d'Ensemble du Pipeline

Le pipeline se déroule en 5 étapes modulaires orchestrées par les scripts d'exécution :

```
[Étape 1 : Inspection] ---> [Étape 2 : DataMix SFT] ---> [Étape 3 : Entraînement SFT]
  (inspect_repo.py)           (prepare_data.py)             (train_sft.py + eval_local.py)
                                                                    |
                                                                    v
[Étape 5 : Export & Éval] <----------------------------- [Étape 4 : RL GRPO & Distill]
 (export_submission.py)                                       (train_rl_grpo.py)
```

---

## 3. Détail des Modules : Théorie vs Code Réel

### Étape 1 : Inspection du Dépôt et Configuration Matérielle
* **Fichiers** : `post_training/inspect_repo.py` et `post_training/training_tracker.py`
* **Objectif** : Valider les poids initiaux de `base_model/model.safetensors` et activer l'accélération matérielle NVIDIA.

#### Implémentation clé :
1. **Détection matérielle et TF32 (`training_tracker.py:configure_nvidia_gpu`)** :
   * Détecte le GPU NVIDIA (ex: RTX 4060) via `torch.cuda.is_available()`.
   * Active le mode TensorFloat-32 (`allow_tf32 = True`) et `cudnn.benchmark = True`.
   * Sélectionne automatiquement le format `bfloat16` pour le calcul mixte (AMP), évitant tout risque d'overflow d'exposant présent en `float16`.
2. **Vérification d'intégrité (`inspect_repo.py:inspect_all`)** :
   * Charge le modèle via `chess_hw.load_model`.
   * Vérifie la structure des 52 tenseurs via `chess_hw.checks.check_safetensors_file` et valide le hash SHA256 du modèle de départ (`81d876...`).
3. **Résolution du moteur Stockfish (`inspect_repo.py:resolve_stockfish_path`)** :
   * Recherche un binaire universel Stockfish local dans `post_training/bin/` pour servir d'oracle et de professeur de renforcement.

---

### Étape 2 : Construction du DataMix SFT à 3 Piliers
* **Fichier** : `post_training/prepare_data.py`
* **Objectif** : Générer un dataset structuré combinant vision globale, défense tactique et apprentissage explicite des règles d'échecs.

#### Fondements théoriques :
Un fine-tuning par simple imitation de parties humaines génère deux écueils majeurs :
1. *Covariate Shift* : face à un moteur alpha-bêta, le modèle sort de sa trajectoire de confort et hallucine des coups illégaux.
2. *Sous-pondération des coups critiques* : les coups forcés sous échec représentent < 5% des coups d'une partie classique mais coûtent la partie s'ils sont ratés.

Pour résoudre cela, nous implémentons un **DataMix à 3 piliers** :

$$\mathcal{D} = \mathcal{D}_{\text{PV}} \cup \mathcal{D}_{\text{Tactique}} \cup \mathcal{D}_{\text{Légalité}}$$

#### Fonctions précises dans `prepare_data.py` :
1. **Pilier 1 (40% — Trajectoires & Variantes Principales)** :
   * Fonctions : `fetch_hf_games`, `_worker_process_task` (`task_type="hf_game"` / `"pv_game"`), `_encode_annotated_game` (`pillar=0`)
   * Génère des séquences complètes jusqu'à 250 demi-coups issues de Lichess ou du livre d'ouvertures officiel (`chess_hw/openings.txt.gz`) déroulées avec Stockfish / Engine.
2. **Pilier 2 (40% — Positions Tactiques & Menaces Réétiquetées)** :
   * Fonctions : `_worker_process_task` (`task_type="onpolicy_tactical"`), `_is_tactical`, `_query_teacher`, `_encode_annotated_game` (`pillar=1`)
   * Analyse les demi-coups de captures, promotions et échecs (`board.is_check()`). Réétiquette le coup cible par l'oracle Stockfish (`_query_teacher`) et assigne un poids de gradient élevé :
     $$w_t = 2.5$$
3. **Pilier 3 (20% — Masque Explicite de Légalité)** :
   * Fonctions : `build_sft_dataset` et `_encode_annotated_game` (`pillar=2`)
   * Pour chaque position en milieu et fin de partie, extrait l'ensemble exact des coups légaux $\mathcal{L}_t \subset \{3, \dots, 1970\}$ sous forme de masque compact (`legal_packed` et `legal_offsets`).
4. **Ensemble de Validation Contrôlé** :
   * Fonctions : `_worker_build_val_position` et `load_or_create_val_positions` (`eval_local.py`)
   * Génère 1 000 positions stratifiées (ouvertures, milieux de partie, échecs tactiques, finales) avec évaluation Stockfish en centipawns pour mesurer rigoureusement l'ACPL (*Average Centipawn Loss*).

---

### Étape 3 : Entraînement SFT Hybride & DAgger
* **Fichiers** : `post_training/train_sft.py` et `post_training/eval_local.py`
* **Objectif** : Maximiser la légalité et la pertinence positionnelle sans surapprentissage.

#### Fondement théorique : Fonction de Perte Hybride
La fonction de perte appliquée à chaque position $t$ intègre 4 composantes complémentaires :

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \alpha \cdot \mathcal{L}_{\text{Uniform}} + \lambda_{\text{illegal}} \cdot \mathcal{L}_{\text{Penalty}} + \beta_{\text{ref}} \cdot \mathcal{L}_{\text{KL-Ref}}$$

1. **Cross-Entropy Pondérée** :
   $$\mathcal{L}_{\text{CE}} = - w_t \cdot \log p_\theta(y_t^* \mid s_t)$$
2. **Lissage Uniforme sur les Coups Légaux** :
   Pour éviter que le modèle n'écrase tous les coups légaux secondaires au profit d'un seul token (ce qui détruirait le Top-2 et Top-3) :
   $$\mathcal{L}_{\text{Uniform}} = \text{KL}\left(\text{Uniform}(\mathcal{L}_t) \parallel p_\theta(\cdot \mid s_t)\right) = -\frac{1}{|\mathcal{L}_t|} \sum_{m \in \mathcal{L}_t} \log p_\theta(m \mid s_t) - \log |\mathcal{L}_t|$$
3. **Pénalité de Fuite de Masse Illégale** :
   Punit directement la somme des probabilités allouées aux tokens interdits :
   $$\mathcal{L}_{\text{Penalty}} = -\log \left(\sum_{m \in \mathcal{L}_t} p_\theta(m \mid s_t)\right) + \sum_{m \notin \mathcal{L}_t} p_\theta(m \mid s_t)$$
4. **Ancrage KL au Modèle de Base** :
   Empêche la divergence catastrophique des représentations pré-entraînées.

#### Fonctions précises dans `train_sft.py` :
* `compute_hybrid_sft_loss(logits, target_ids, weights, legal_mask, valid_mask, ...)` :
  Implémente vectoriellement les 4 termes de perte avec masquage booléen GPU.
* `collect_onpolicy_refresh_samples(model_path, n_games, ...)` :
  Implémente l'algorithme **DAgger** (*Dataset Aggregation*) : fait jouer le modèle en cours d'entraînement contre l'Engine officiel et réinjecte à chaque époque les positions réelles rencontrées avec la correction de l'oracle Stockfish.
* `build_optimizer_and_scheduler(model, lr, weight_decay, total_steps, ...)` :
  Configure l'optimiseur `AdamW` ($\beta_1=0.9, \beta_2=0.95$) avec un *cosine annealing schedule* et 8% de warmup linéaire. Exclut du weight-decay les biais et LayerNorms.
* `evaluate_model_on_positions(model, val_positions, ...)` dans `eval_local.py` :
  Évalue le checkpoint à chaque époque sur les 1 000 positions de test et vérifie le **SFT Gate** :
  $$\text{Top-1 Legal} \ge 92\% \quad \text{et} \quad \text{Top-3 Legal} \ge 98\%$$

---

### Étape 4 : Apprentissage par Renforcement (GRPO & Q-Distillation)
* **Fichier** : `post_training/train_rl_grpo.py`
* **Objectif** : Optimiser directement le score de partie sans modèle de valeur séparé (critique) en s'appuyant sur l'oracle Stockfish.

#### Fondements théoriques :
1. **Single-Step Group Relative Policy Optimization (GRPO)** :
   Au lieu d'entraîner un réseau critique lourd (impossible sous la barre de 1M de paramètres), pour chaque position $s$, on échantillonne un groupe de $G = 8$ actions candidates :
   $$A = \{a_1, a_2, \dots, a_G\}$$
   L'oracle Stockfish attribue une récompense scalaire $R_i$ à chaque candidat :
   * Coup illégal : $R = -2.0$
   * Mat trouvé : $R = +2.0$
   * Coup légal : $R_i = -0.5 + 1.5 \exp\left(\frac{\Delta \text{cp}_i}{\tau_{\text{reward}}}\right) \in [-0.5, 1.0]$ où $\Delta \text{cp}_i = \text{cp}_i - \text{cp}^* \le 0$.
   
   L'avantage relatif intra-groupe normalisé s'écrit :
   $$\hat{A}_i = \frac{R_i - \text{mean}(R)}{\text{std}(R) + \epsilon}$$
   La perte de politique utilise le surrogate clippé standard (style PPO/GRPO) :
   $$\mathcal{L}_{\text{GRPO}} = - \frac{1}{G} \sum_{i=1}^G \min \left( r_i(\theta) \hat{A}_i, \, \text{clip}(r_i(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i \right)$$
   avec $r_i(\theta) = \frac{\pi_\theta(a_i \mid s)}{\pi_{\text{old}}(a_i \mid s)}$.

2. **Distillation Boltzmann de Q-valeurs (Zero-Variance)** :
   Pour guider le modèle analytiquement, nous calculons la distribution de Boltzmann idéale sur l'ensemble des coups légaux $\mathcal{L}$ :
   $$q^*(a \mid s) = \frac{\exp(Q_{\text{SF}}(s, a) / \tau_Q)}{\sum_{a' \in \mathcal{L}} \exp(Q_{\text{SF}}(s, a') / \tau_Q)}$$
   La perte de distillation minimise la divergence KL vers cette cible :
   $$\mathcal{L}_{\text{Q-Distill}} = \text{KL}\left( q^*(\cdot \mid s) \parallel \pi_\theta(\cdot \mid s) \right)$$

#### Fonctions précises dans `train_rl_grpo.py` :
* `_init_rl_worker` & `_evaluate_all_legal_moves_cp(board)` :
  Workers parallèles multiprocessing maintenant des instances permanentes du moteur UCI Stockfish en mémoire pour évaluer simultanément les coups légaux en centipawns.
* `sample_group_candidates(logits_last, group_size=8, strategy='top3_plus_sample')` :
  Échantillonne les $G=8$ candidats par position en garantissant la présence des 3 meilleurs choix déterministes (Top-3 glouton) complétés par un échantillonnage multinomial exploratoire.
* `run_rl_grpo_training(args)` :
  Boucle d'entraînement globale calculant les rollouts, appliquant les gradients clippés et sélectionnant le meilleur checkpoint selon le score composite de validation.

---

### Étape 5 : Export, Vérification Structurelle et Match de Soumission
* **Fichier** : `post_training/export_submission.py`
* **Objectif** : Générer le checkpoint final scellé, certifier sa conformité matérielle et mesurer le gain Elo contre le modèle initial.

#### Fonctions précises dans `export_submission.py` :
1. **Sauvegarde et Validation Safetensors (`export_and_verify`)** :
   * Charge le meilleur checkpoint (`checkpoints/sft_best` ou `checkpoints/rl_best`) via `load_any_checkpoint`.
   * Enregistre le modèle nettoyé au format float32 dans `post_training/final_model/model.safetensors` via `chess_hw.save_model`.
   * Exécute `chess_hw.checks.check_safetensors_file` pour certifier l'absence de NaN, de format bf16/fp16 non supporté ou de tenseurs discontinus en mémoire.
2. **Mesure de l'Écart Paramétrique** :
   * Calcule la norme L2 de modification par couche par rapport aux poids initiaux (`base_model`) :
     $$\Delta L_2 = \|\theta_{\text{final}} - \theta_{\text{base}}\|_2 = 2.88$$
     Cela atteste mathématiquement que les poids ont été adaptés tout en restant dans le bassin de convergence du modèle pré-entraîné.
3. **Match Tête-à-Tête vs Base Model (`evaluate_head_to_head`)** :
   * Fait jouer 50 parties en miroir (25 ouvertures $\times$ 2 couleurs blanc/noir) entre le modèle entraîné et le modèle de base, en comptabilisant les victoires, nulles, défaites et les taux de légalité respectifs.
4. **Vérification Officielle de Soumission** :
   * Lance automatiquement `python submit.py --model post_training/final_model --writeup post_training/WRITEUP.md --dry_run` pour garantir que l'archive chiffrée finale est prête pour le serveur Hugging Face.

---

## 4. Synthèse des Résultats Obtenus

Les métriques extraites des journaux d'exécution ([logs/sft_val.csv](file:///c:/Users/aimin/Documents/ENS%20MVA/Fondation_of_LLM/HW_optional/chess-post-training/post_training/logs/sft_val.csv) et [final_model/verification_report.json](file:///c:/Users/aimin/Documents/ENS%20MVA/Fondation_of_LLM/HW_optional/chess-post-training/post_training/final_model/verification_report.json)) valident l'efficacité du post-entraînement :

| Indicateur | Base Model | Modèle Final Post-Entraîné | Interprétation |
| :--- | :---: | :---: | :--- |
| **Validation Loss** | $2.885$ | **$2.432$** | Réduction de $0.45$ nat de perplexité |
| **Top-1 Légalité** | $95.10\%$ | **$99.00\%$** | Le 1er coup prédit est quasiment toujours légal |
| **Top-3 Légalité** | $98.80\%$ | **$99.80\%$** | Dépassement net du seuil éliminatoire de 98% |
| **Taux de Coup Aléatoire** | $1.20\%$ | **$0.20\%$** | Chute drastique des situations de blocage (divisé par 6) |
| **Stockfish Top-1 Accuracy** | $30.40\%$ | **$34.90\%$** | $+4.5\%$ d'accord avec le meilleur coup absolu |
| **Résultat Engine Niveau 0** | ~50% | **$63.0\%$** (13 V, 37 N, 0 D) | Aucune défaite enregistrée face au niveau 0 |

Le checkpoint final validé se trouve dans `post_training/final_model/model.safetensors` et est accompagné de sa note technique `post_training/WRITEUP.md`.
