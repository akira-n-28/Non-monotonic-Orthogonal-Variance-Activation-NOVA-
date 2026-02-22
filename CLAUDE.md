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

### Da Fare
* **Nano-GPT** (TinyShakespeare): risultati preliminari nel paper (NOVA 1.6949 vs GELU 1.7344), script da aggiornare con confronto multi-attivazione.
* **PINN Burgers 1D**: risultati preliminari nel paper (NOVA 0.00027 vs GELU 0.00353), script da aggiornare.
* **DDPM Fashion-MNIST**: risultati preliminari nel paper (NOVA 0.0382 vs GELU 0.0372), script da aggiornare.
* **Scaling ViT**: esperimenti a scala maggiore per verificare se il vantaggio NOVA si mantiene.

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
