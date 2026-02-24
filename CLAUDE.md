# **Progetto NOVA: Non-monotonic Orthogonal Variance Activation**

Questo file definisce le regole assolute, il contesto e le linee guida operative per qualsiasi contributo, modifica o analisi all'interno di questa repository. Leggi attentamente prima di iniziare qualsiasi sessione.

## **1\. Contesto del Progetto**

NOVA è una nuova funzione di attivazione neurale progettata per bilanciare l'auto-regolarizzazione geometrica e la propagazione della varianza.

**Formula Matematica:** f(x) \= x \* sigmoid(βx) \- x / (1 \+ (βx)^2)

Dove β (beta) è un parametro apprendibile.

## **2\. Regole Assolute sull'Integrità dei Dati**

* **NON INVENTARE MAI DATI.** Qualsiasi metrica, loss, accuracy o tempo di calcolo menzionato deve derivare da un log reale presente nella cartella results/.
* **NO STATISTICHE FALSE.** Poiché gli esperimenti attuali sono single-run (singolo seed), è severamente vietato inventare o calcolare deviazioni standard, medie su run multiple o p-value.
* Se un dato o un test manca, dichiaralo esplicitamente come "Da verificare in lavori futuri".

## **3\. Stato Tecnico e Kernel CUDA**

Il Kernel CUDA fuso (forward/backward) è implementato con le seguenti caratteristiche:

* Il backward pass calcola i gradienti sia per l'input x sia per il parametro β (riduzione via `torch::sum` sul gradiente per-elemento).
* Supporto nativo per FP32, FP64, FP16 (Half) e BF16 tramite `AT_DISPATCH_FLOATING_TYPES_AND2`. Le operazioni intermedie sono eseguite in float32 per stabilità numerica.
* β è registrato come `nn.Parameter` apprendibile in `NOVAFusedCUDA`.

## **4\. Linee Guida per il Paper LaTeX**

* **Hardware:** Dichiarare sempre l'utilizzo di GPU NVIDIA T4 (hardware consumer/Kaggle, non HPC).
* **Linguaggio Cauto:** Usa "osserviamo che", "suggerisce", "risultati preliminari". Non usare "dimostriamo", "stato dell'arte" o "statisticamente significativo" per run singole.
* Il paper è attualmente un *Preprint / Work in Progress*.

## **5\. Struttura del Progetto**

* /src: Implementazione di NOVA (Python ed estensione CUDA).
* /experiments: Script per testare NOVA nei vari domini (ViT, Nano-GPT, DDPM, PINN).
* /results: Log grezzi degli esperimenti e immagini generate (LA SOURCE OF TRUTH).
* /paper: Codice sorgente LaTeX del preprint.

## **6\. Stato Esperimenti**

### Completati
* **ViT CIFAR-100** (`experiments/vit_cifar100.py`): Mini-ViT (4L, 256d, 4h) su CIFAR-100, 100 epoche, FP16, 2×T4 Kaggle.
  - Confronto 5 attivazioni: NOVA (56.75%) > GELU (54.67%) > ReLU (54.55%) > SiLU (53.84%) > Mish (53.25%)
  - Config: AdamW lr=3e-3, wd=0.05, warmup 10ep + cosine, batch 1024, label smoothing 0.1
  - Log in `results/vit_cifar100_{activation}_20260222.json`

* **Scaling ViT** (`experiments/vit_scaling.py`): Studio di scaling NOVA vs GELU su 3 varianti ViT, CIFAR-100, 100 epoche, FP16, 2×T4 Kaggle.
  - Tiny (4L, 256d, 4h, 3.2M params): NOVA 56.71% vs GELU 54.43% (+2.28)
  - Small (6L, 384d, 6h, 10.7M params): NOVA 59.37% vs GELU 55.48% (+3.89)
  - Base (8L, 512d, 8h, 25.3M params): NOVA 55.22% vs GELU 51.15% (+4.07)
  - Overfitting marcato a tutte le scale; accuracy in validazione degrada da Small a Base
  - Log in `results/vit_scaling_{scale}_{activation}_20260222*.json`

* **Scaling ViT v2 — Anti-Overfitting** (`experiments/vit_scaling_v2.py`): Regolarizzazione DeiT-style (RandAugment, CutMix+Mixup, DropPath).
  - Tiny (3.2M): NOVA 51.72% vs GELU 45.12% (+6.60) — sotto-adattamento, regolarizzazione troppo aggressiva
  - Small (10.7M): NOVA 62.66% vs GELU 54.31% (+8.35) — miglior risultato assoluto (+3.29 vs v1)
  - Base (25.3M): NOVA 60.31% vs GELU 53.14% (+7.17) — recupero significativo (+5.09 vs v1)
  - Overfitting eliminato; vantaggio NOVA amplificato (da +2-4 pp in v1 a +6-8 pp in v2)
  - β converge a ~0.45 per tutte le scale
  - Log in `results/vit_scaling_v2_{scale}_{activation}_20260222*.json`
  - Plot in `results/plot_v2_*.png`

* **ViT su Tiny-ImageNet-200** (`experiments/vit_tinyimagenet.py`): Validazione cross-dataset su Tiny-ImageNet-200 (200 classi, 64×64).
  - Architettura: stessa famiglia ViT (Tiny/Small/Base), patch_size=8 (→ 64 patch)
  - Regolarizzazione calibrata per scala: Tiny (RandAug mag 5, CutMix/Mixup prob 0.5, DropPath 0.05), Small/Base (DeiT-style piena)
  - Config: Tiny batch 512 lr=2e-3, Small batch 256 lr=1e-3, Base batch 128 lr=5e-4
  - Tiny (3.3M): NOVA 47.71% vs GELU 45.96% (+1.75) | Top-5: 72.52% vs 71.32%
  - Small (10.8M): NOVA 51.01% vs GELU 46.43% (+4.58) | Top-5: 75.49% vs 72.18%
  - Base (25.5M): NOVA 50.75% vs GELU 44.70% (+6.05) | Top-5: 75.32% vs 70.15%
  - Vantaggio monotonamente crescente con la scala, conferma cross-dataset
  - β converge a ~0.54-0.58 (più alto di CIFAR-100 v2, adattamento a dataset più complesso)
  - Log in `results/vit_tinyimagenet_{scale}_{activation}_20260223*.json`
  - Plot in `results/plot_tinyimagenet_*.png`

* **ConvNeXt CIFAR-100** (`experiments/convnext_cifar100.py`): Generalizzazione architetturale su CNN pura (no attention).
  - ConvNeXt (Liu et al. 2022) adattato per CIFAR-100 (32×32), stem stride-1, Layer Scale
  - Tiny (dims [40,80,160,320], depths [2,2,6,2], 3.5M): NOVA 65.88% vs GELU 65.74% (+0.14)
  - Small (dims [64,128,256,512], depths [3,3,6,3], 11.0M): NOVA 75.01% vs GELU 74.36% (+0.65)
  - Base (dims [96,192,384,768], depths [3,3,9,3], 28.1M): NOVA 79.03% vs GELU 78.07% (+0.96)
  - Vantaggio NOVA presente ma più contenuto rispetto a ViT (+0.14-0.96 vs +6.60-7.17)
  - β converge a ~0.23-0.30 (molto più basso di ViT ~0.45), NOVA si avvicina a SiLU in contesto CNN
  - Config: Tiny batch 2048 lr=4e-3, Small batch 1024 lr=3e-3, Base batch 512 lr=2e-3
  - Log in `results/convnext_cifar100_{scale}_{activation}_20260223*.json`
  - Plot in `results/plot_convnext_*.png`

* **ConvNeXt CIFAR-100 v2 — Iperparametri Ottimizzati** (`experiments/convnext_cifar100.py`): Stessa architettura, batch size e learning rate ridotti.
  - Config v2: Tiny batch 256 lr=1e-3, Small batch 256 lr=1e-3, Base batch 128 lr=5e-4
  - Tiny (3.5M): NOVA 70.97% vs GELU 69.45% (+1.52) — miglioramento assoluto +5pp vs v1, delta NOVA 10× più grande
  - Small (11.0M): NOVA 76.39% vs GELU 75.46% (+0.93) — miglioramento assoluto +1.4pp vs v1
  - Base (28.1M): NOVA 77.99% vs GELU 76.62% (+1.37) — accuracy assoluta leggermente inferiore a v1, ma delta NOVA sale
  - Vantaggio NOVA amplificato a tutte le scale (da +0.14-0.96 in v1 a +0.93-1.52 in v2)
  - Batch size più piccolo favorisce Tiny (+5pp) ma penalizza Base (-1pp assoluto), suggerendo che gli iperparametri ottimali vadano calibrati per scala
  - Log in `results/convnext_cifar100_v2_{scale}_{activation}_*.json`

* **DiT DDPM su CIFAR-10 v2** (`experiments/dit_cifar10_v2.py`): Diffusion Transformer su CIFAR-10, solo scala Base.
  - **Motivazione:** Il risultato U-Net (NOVA 0.0382 > GELU 0.0372, GELU vince) è spiegato come "Smoothness Trade-off": la U-Net usa BatchNorm, che entra in conflitto con la non-monotonia di NOVA. Il DiT (Peebles & Xie, 2023) usa LayerNorm + AdaLN — contesto dove NOVA eccelle nei ViT. L'ipotesi era che NOVA recuperi con backbone Transformer.
  - **Architettura:** DiT-Base (8L, 384d, 6h, MLP×4, ~22M params), AdaLN-Zero conditioning, patch 4×4
  - **Diffusion config:** Cosine schedule, T=1000, MSE ε-prediction, 400 epoche, DDIM sampler (250 step, η=0)
  - **Training config:** AdamW lr=3e-4, wd=0.01, warmup 10ep + cosine decay, FP16+GradScaler, EMA decay 0.9995, grad clip 1.0, batch 128
  - **Miglioramenti v2 rispetto a v1:** 400 epoche (da 100), EMA 0.9995 (da 0.9999), DDIM sampler (da DDPM), weight_decay 0.01, RandomCrop augmentation, FID ogni 25 epoche
  - **Risultati (DiT-Base, ~22M params):**
    - NOVA: best val loss 0.054673, best FID 21.74 (ep375), best IS 4.10
    - GELU: best val loss 0.054585, best FID 21.45 (ep400), best IS 4.11
    - Le due attivazioni sono sostanzialmente equivalenti su DiT-Base
    - Il divario FID è minimo (0.29 punti, entro la varianza di run singola)
  - **Evoluzione FID:** ep25: 88/86 → ep100: 32/31 → ep200: 25/25 → ep300: 23/22 → ep400: 22/21 (NOVA/GELU)
  - **β NOVA:** converge a ~0.756 (più alto di ViT ~0.45 e ConvNeXt ~0.25)
  - **Interpretazione:** L'ipotesi del recupero completo su DiT non è confermata — NOVA e GELU sono paragonabili, non c'è vantaggio significativo per nessuna delle due. Questo suggerisce che nel contesto generativo (ε-prediction) il beneficio della sacca non-monotona è neutro: non danneggia (come nella U-Net con BN) ma non aiuta (come nei ViT classificativi).
  - Log in `results/dit_cifar10_base_{activation}_20260224*.json`

### Da Fare
* **Scaling ViT v3 — Regolarizzazione Calibrata** (`experiments/vit_scaling_v3.py`): Corregge il sotto-adattamento Tiny di v2 calibrando la regolarizzazione per scala.
  - Tiny: RandAugment mag 5 (era 9), CutMix/Mixup prob 0.5 (era 1.0), DropPath 0.05 (era 0.1)
  - Small/Base: invariati rispetto a v2
  - Plot con confronto v1/v2/v3
  - Uso: `python vit_scaling_v3.py` (full run) oppure `python vit_scaling_v3.py --plot-only`
* **Nano-GPT** (`experiments/nanogpt_shakespeare.py`): Scaling study su language modeling autoregressivo, TinyShakespeare char-level.
  - **Dataset:** TinyShakespeare (~1MB, char-level, vocab ~65 token), split 90/10
  - **Architettura:** Decoder-only Transformer (pre-LN, causal attention, weight tying)
    - Tiny (4L, 256d, 4h, MLP×4, ~3M params)
    - Small (6L, 384d, 6h, MLP×4, ~10M params)
    - Base (8L, 512d, 8h, MLP×4, ~25M params)
  - **Training:** 5000 iterazioni, batch 64, context 256, AdamW (betas 0.9/0.95), warmup 500 iter + cosine decay, FP16, grad clip 1.0
    - Tiny lr=1e-3, Small lr=6e-4, Base lr=3e-4, weight_decay=0.01
  - **Attivazioni:** NOVA, GELU, SiLU, Mish, ReLU (5 confronti)
  - **Metriche:** Val loss (CE), perplexity (exp(val_loss)), β evolution, campioni di testo generato
  - **Eval:** ogni 250 iter (200 batch random), campioni testo ogni 1000 iter
  - **Plot (4):**
    1. Training curves per scala: val loss vs iterazione
    2. Scaling curve: best val loss vs parametri
    3. Perplexity scaling: best perplexity vs parametri
    4. β evolution: convergenza di β per scala (NOVA)
  - **Output:**
    - Log: `results/nanogpt_{scale}_{activation}_{timestamp}.json`
    - Plot: `results/plot_nanogpt_*.png`
  - Uso: `python nanogpt_shakespeare.py` (full run) oppure `python nanogpt_shakespeare.py --scale small --activation nova --gpu 0` oppure `python nanogpt_shakespeare.py --plot-only`
  - Risultati preliminari precedenti (solo Small, solo NOVA vs GELU, 1000 iter): NOVA 1.6949 vs GELU 1.7344
* **PINN Burgers 1D**: risultati preliminari nel paper (NOVA 0.00027 vs GELU 0.00353), script da aggiornare.
* **DDPM Fashion-MNIST**: risultati preliminari nel paper (NOVA 0.0382 vs GELU 0.0372), script da aggiornare.

## **7\. Workflow per Nuovi Esperimenti**

Usa sempre il "Plan Mode" prima di task complessi o di scrivere nuovi esperimenti.

Ogni nuovo script sperimentale deve contenere:

* set\_seed(42) per riproducibilità.
* Salvataggio automatico dei log e dei grafici nella cartella results/ con timestamp.
* Mixed Precision (FP16) con calcoli critici in FP32 per le T4.
* Pre-download dataset e pre-compilazione kernel CUDA nel launcher (evita race condition su multi-GPU).
* Supporto multi-attivazione (almeno NOVA, GELU, ReLU, SiLU, Mish).

## **8\. Convenzioni Codice**

* Attivazioni supportate: `nova`, `gelu`, `silu`, `mish`, `relu`.
* NOVA viene creata tramite `make_nova(beta=1.0)` che preferisce CUDA e fallback Python.
* Il launcher multi-GPU lancia coppie di esperimenti in parallelo su 2 GPU.
* Formato log: JSON con campi obbligatori (esperimento, data, hardware, seed, dataset, modello, obiettivo, metriche, tempo_totale_sec).
