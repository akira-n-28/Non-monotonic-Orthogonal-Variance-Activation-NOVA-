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

### Da Fare
* **Nano-GPT** (TinyShakespeare): risultati preliminari nel paper (NOVA 1.6949 vs GELU 1.7344), script da aggiornare con confronto multi-attivazione.
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
