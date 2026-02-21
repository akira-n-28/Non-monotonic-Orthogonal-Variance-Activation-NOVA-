# **Checklist di Integrità e Workflow**

Consulta questa checklist prima di iniziare a programmare, prima di effettuare un commit e prima di aggiornare il documento LaTeX.

## **Pre-Sessione (Pianificazione)**

* \[ \] L'obiettivo del task odierno è chiaro ed esplorato tramite "Plan Mode" (se assistiti da AI)?  
* \[ \] Se si aggiunge un nuovo esperimento, il template di log è pre-impostato per salvare in results/?  
* \[ \] L'hardware target (NVIDIA T4) è preso in considerazione per le stime di tempo/memoria?

## **Pre-Commit (Integrità del Codice)**

* \[ \] Il diff è stato analizzato riga per riga?  
* \[ \] Non sono stati alterati o mascherati i bug noti del kernel CUDA (es. derivata di β, BF16)?  
* \[ \] I risultati nei commenti del codice corrispondono esattamente all'ultimo output del terminale?  
* \[ \] Il codice fissa i seed per PyTorch/Numpy/Random per garantire replicabilità base?

## **Pre-Aggiornamento Paper LaTeX**

* \[ \] **INTEGRITÀ:** Ogni singolo numero (accuracy, loss, latenza) aggiunto al paper esiste nei file log in results/?  
* \[ \] **LINGUAGGIO CAUTO:** Hai rimosso parole come "dimostra", "statisticamente significativo", "dominio assoluto"?  
* \[ \] **NO FABBRICAZIONI:** Hai evitato di inserire deviazioni standard (±) o medie per esperimenti single-run?  
* \[ \] Le limitazioni dell'esperimento (singolo seed, hardware limitato, instabilità di β) sono state dichiarate chiaramente nella relativa sezione?