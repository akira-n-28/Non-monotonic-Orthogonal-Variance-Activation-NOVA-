# **NOVA (Non-monotonic Orthogonal Variance Activation)**

**Preprint / Work in Progress**

NOVA è una funzione di attivazione ibrida (razionale-esponenziale) sperimentale progettata per ottimizzare la propagazione della varianza profonda e fornire un'Hessiana ad alta espressività per operatori differenziali.

**Formula matematica:**

**![][image1]***Nota: Questa repository contiene il codice sorgente per l'attivazione (incluso un kernel CUDA custom preliminare) e gli script per riprodurre gli esperimenti esplorativi descritti nel nostro preprint in stesura.*

## **Risultati Preliminari (Esplorativi)**

**⚠️ ATTENZIONE:** Tutti i risultati attuali sono basati su **singole run con singolo seed**, eseguiti su una singola GPU **NVIDIA T4**. Rappresentano trend qualitativi e non evidenze statisticamente definitive.

* **Vision Transformers (ViT-Mini su CIFAR-10, 10 epoche):** \* NOVA: 68.31% | GELU: 67.54%  
* **LLM Autoregressivo (Nano-GPT su TinyShakespeare, 1000 iter, Val Loss):**  
  * NOVA: 1.6949 | GELU: 1.7344  
* **Scientific ML / PINN (Equazione di Burgers 1D, PDE Residual Loss):**  
  * NOVA: 0.00027 | GELU: 0.00353 *(Riduzione dell'errore di un fattore \~13x)*  
* **Modelli di Diffusione (U-Net DDPM su Fashion-MNIST, Denoising Loss):**  
  * NOVA: 0.0382 | GELU: 0.0372 *(GELU mantiene prestazioni superiori)*

## **Limitazioni e Bug Noti**

L'integrità scientifica richiede di evidenziare i limiti attuali del nostro lavoro:

* **Nessuna replicazione statistica:** Gli intervalli di confidenza saranno affrontati in lavori futuri con ![][image2] run.  
* **Scala ridotta:** Gli esperimenti testano architetture piccole su dataset contenuti; la validazione su LLM/ViT massicci è pending.  
* **Instabilità di β:** Se lasciato come parametro apprendibile senza un attento warmup dell'ottimizzatore, ![][image3] tende a divergere. Attualmente è fissato a 1.0 in gran parte dei benchmark.  
* **Limiti Kernel CUDA:** La fusione C++ velocizza il training ma attualmente non calcola il gradiente per ![][image3] nel backward pass e non è ottimizzata per BF16.  
* **Smoothness Trade-off:** NOVA soffre marginalmente rispetto a GELU nei Modelli di Diffusione (DDPM) a causa dell'effetto delle micro-fluttuazioni nell'integrazione di campi vettoriali puri.

## **Installazione e Uso**

git clone \[https://github.com/your-username/nova-activation.git\](https://github.com/your-username/nova-activation.git)  
cd nova-activation  
pip install \-r requirements.txt

Per utilizzare il kernel CUDA fuso (necessita di compilatore NVCC disponibile):

from src.nova\_cuda import compile\_nova\_cuda, NOVAFusedCUDA  
nova\_ext \= compile\_nova\_cuda()  
activation \= NOVAFusedCUDA(nova\_ext)

## **Come Replicare gli Esperimenti**

Puoi lanciare gli script nella cartella /experiments. Se intendi aggiungere nuovi test, ti chiediamo di adottare il seguente standard obbligatorio per l'header del file log:

\# ESPERIMENTO: \[nome\_esperimento\]  
\# Data: \[YYYY-MM-DD\]  
\# Hardware: NVIDIA T4  
\# Seed: \[numero\_seed\]  
\# Dataset: \[nome\_dataset\]  
\# Modello: \[descrizione\_architettura\]  
\# Obiettivo: \[cosa si vuole misurare\]  
\# Risultati attesi: \[ipotesi teorica\]

Assicurati che lo script richiami torch.manual\_seed(SEED) e che tutti i log e le figure generati vengano salvati automaticamente nella directory results/.

[image1]: <data:image/png;base64,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>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADUAAAAZCAYAAACRiGY9AAACuklEQVR4Xu2WPWgUURDH9zgFQUVEjyP39e4O5RAshAPFkNIuRDTYKVgcaOMHNrFQxMYqYHFYhYgoiGKhjUmhAYMRPEsL0dpCUJA0Wqhg/M3t27u5l7vc7sYIwv5heLsz//d25s2beet5CRIk2DCUy+UDxpiFUqn0kfED43nUKc1BfwXeXcaZQGSe5mw0KpVKliHtqOW9x9c2CoVCnkBOIBdxdgVZLhaL+xUlhW4U+wXG74zXGMez2exWxekB9uMyx1vtRCxkMpltrLeI/FQbu4B8Y3O3uPwOIDQhXLeBvbc700Eul9uNfp/WDUM+n9/FnM/ITWTEtYdFEJQ6LQ1Z2+X1oFarbYf4VDLEuGwDO6c5ZGgM2al1YSAZFSeQT8hjr99xGYIgKL4/4doGQjKAzPO4iXHWBtXSHHbprH6Pinq9vpk1jrHuaxPxaMYN6iQT7tjnw8gP5LeiSLCP1HtsENghG9g7CdK190MQFCfpKGMTmcHfUwPrWo4UpHmj6oXnSQlKArScIzjwrDtr/bD1Ni2NR7Lo2h1Is7oHdypQ0OD2ovvCOgVNbANiHeNzXS/VanUHuhZyi9cUtss831DTYoMGVBMHjV9jDdceFqoj9tR+G1IrGJquXsjyYds8nki2XE5EpFln1NhjFyI7wyDZuy8iz/0Mk1opQFdBVgjmIeOS3GcuJwxUg5Bg5AhH7n74cIb5X5GrWm9b/KJkraO0d88LM+D+kaCszJXXuuQGQAq55P+p3JZNcu1hoI6Z+PE20NurQi7gWc1P4+hLlC2ysMfrs4PYHiC/cGzMta0F1r3EvMbA7hQR1OFB1ntl/RRInU9JoJ1jbLptW6IPZK67jA/hMfmNZNS1/Wvgy7jx/07kj2LJ+N152uX9d5CuzCk4zUZPuL9xCeLAFvdIGLE/pqvqOUGCBAn+Ov4ADRy2VZzfdRsAAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAZCAYAAAAFbs/PAAABOUlEQVR4Xt2Svy4EURTGr+xKbCKhMJKd/3+iEIlmKlklhUahEIkXoPUOXkBUOgoiS+EVdB5AFCjoNQqNBL8zmZF7z+w+AF/yZe78znfu3DkzxvwtpWk6myRJn+uUrrVE6DaO4zuulzS9l2U5qTO/IrAVBEHY3IdhuEDjnp1pCj3CQ3Zetzn3OQ2nNqtEeAVfSaPNoyjapGHHZpUIH+Fdm+V5PiPv43netM2NgPo4SwT2WX/gR/yEr52wSHbGh5qzwSr8U3NpOMEbI3gfv7DsOgWOcSPTcKCpGhbxW+sdgOejvqhMiNq35vKE9pyNmSB8jL8c6vv+HMd5xYOGZVm2TPAZD2W0dr6ZhHy0e3mXegAP8G3KHScsonhQLztFUczLZJyAUpfAmYZjxWPXaLjQfKzk15VZa/6f9QM3gj0rD789IwAAAABJRU5ErkJggg==>
