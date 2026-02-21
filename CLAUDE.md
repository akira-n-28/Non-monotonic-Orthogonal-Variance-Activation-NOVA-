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

* **Hardware:** Dichiarare sempre l'utilizzo di una singola GPU NVIDIA T4 (hardware consumer/Colab, non HPC).  
* **Linguaggio Cauto:** Usa "osserviamo che", "suggerisce", "risultati preliminari". Non usare "dimostriamo", "stato dell'arte" o "statisticamente significativo" per run singole.  
* Il paper è attualmente un *Preprint / Work in Progress*.

## **5\. Struttura del Progetto**

* /src: Implementazione di NOVA (Python ed estensione CUDA).  
* /experiments: Script per testare NOVA nei vari domini (ViT, Nano-GPT, DDPM, PINN).  
* /results: Log grezzi degli esperimenti e immagini generate (LA SOURCE OF TRUTH).  
* /paper: Codice sorgente LaTeX del preprint.

## **6\. Workflow per Nuovi Esperimenti**

Usa sempre il "Plan Mode" prima di task complessi o di scrivere nuovi esperimenti.

Ogni nuovo script sperimentale deve contenere:

* set\_seed(42) per riproducibilità.  
* Salvataggio automatico dei log e dei grafici nella cartella results/ con timestamp.