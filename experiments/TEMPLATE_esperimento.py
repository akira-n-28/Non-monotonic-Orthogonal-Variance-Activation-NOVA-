# -*- coding: utf-8 -*-
"""
TEMPLATE ESPERIMENTO NOVA
==========================
Copia questo file e rinominalo per ogni nuovo esperimento.
Segui la struttura senza rimuovere le sezioni obbligatorie.
"""

import os
import time
import json
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

# ==============================================================
# CONFIGURAZIONE ESPERIMENTO (modifica questi parametri)
# ==============================================================
EXPERIMENT_NAME = "nome_esperimento"       # es. "vit_cifar10", "nanogpt_shakespeare"
SEED = 42
DATASET = "nome_dataset"                   # es. "CIFAR-10", "TinyShakespeare"
MODEL_DESCRIPTION = "descrizione_modello"  # es. "ViT-Mini, 4 layers, 256 dim"
OBJECTIVE = "cosa si vuole misurare"       # es. "Test accuracy a 10 epoche"
HARDWARE = "NVIDIA T4"

# ==============================================================
# RIPRODUCIBILITA'
# ==============================================================
def set_seed(seed: int) -> None:
    """Fissa tutti i seed per riproducibilità."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(SEED)

# ==============================================================
# LOGGING AUTOMATICO
# ==============================================================
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILENAME = f"{EXPERIMENT_NAME}_{TIMESTAMP}.json"
LOG_PATH = os.path.join(RESULTS_DIR, LOG_FILENAME)

log_data = {
    "esperimento": EXPERIMENT_NAME,
    "data": datetime.now().strftime("%Y-%m-%d"),
    "hardware": HARDWARE,
    "seed": SEED,
    "dataset": DATASET,
    "modello": MODEL_DESCRIPTION,
    "obiettivo": OBJECTIVE,
    "metriche": {},
    "tempo_totale_sec": None,
}

# ==============================================================
# ESPERIMENTO (scrivi il tuo codice qui)
# ==============================================================
def run_experiment() -> dict:
    """
    Implementa qui la logica dell'esperimento.
    Deve restituire un dizionario di metriche, es:
        {"train_loss": 0.123, "val_accuracy": 0.456}
    """
    metrics = {}

    # TODO: Implementa il tuo esperimento qui
    # 1. Carica il dataset
    # 2. Definisci il modello (con NOVA o baseline)
    # 3. Training loop
    # 4. Valutazione finale
    # 5. Popola metrics con i risultati

    return metrics

# ==============================================================
# MAIN
# ==============================================================
if __name__ == "__main__":
    print(f"{'='*60}")
    print(f" ESPERIMENTO: {EXPERIMENT_NAME}")
    print(f" Data: {log_data['data']}")
    print(f" Hardware: {HARDWARE}")
    print(f" Seed: {SEED}")
    print(f" Dataset: {DATASET}")
    print(f" Modello: {MODEL_DESCRIPTION}")
    print(f" Obiettivo: {OBJECTIVE}")
    print(f"{'='*60}")

    start_time = time.time()
    metrics = run_experiment()
    elapsed = time.time() - start_time

    log_data["metriche"] = metrics
    log_data["tempo_totale_sec"] = round(elapsed, 2)

    with open(LOG_PATH, "w") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    print(f"\nRisultati salvati in: {LOG_PATH}")
    print(f"Tempo totale: {elapsed:.2f}s")
    print(f"Metriche: {json.dumps(metrics, indent=2)}")
