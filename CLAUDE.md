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

Il Kernel CUDA fuso (forward/backward) è implementato ma ha dei **bug noti e limitazioni da rispettare**:

* Il gradiente per il parametro β non è attualmente calcolato nel backward pass C++ (è trattato come costante fissa a 1.0 per default nei test scalati).  
* Manca il supporto nativo per precisione BF16/FP16 ottimizzata.

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