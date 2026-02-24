# results/

Questa cartella è la **unica fonte di verità** (source of truth) per tutti i dati sperimentali del progetto NOVA.

## Regole

- **Nessun dato manuale.** Ogni file in questa cartella deve essere generato automaticamente da uno script in `experiments/`.
- **Nessuna modifica post-hoc.** I log non vanno mai editati dopo la generazione. Se un esperimento va ripetuto, si genera un nuovo file.
- **Nessuna statistica inventata.** Gli esperimenti attuali sono single-run (seed singolo). È vietato aggiungere deviazioni standard, medie su run multiple o p-value.

## Convenzione di Naming

I file di log seguono il formato:

```
{nome_esperimento}_{YYYYMMDD_HHMMSS}.json
```

Esempi:
- `vit_cifar10_20260221_143025.json`
- `nanogpt_shakespeare_20260221_150312.json`

## Struttura Attesa dei Log (JSON)

Ogni file di log deve contenere almeno questi campi:

```json
{
  "esperimento": "nome_esperimento",
  "data": "YYYY-MM-DD",
  "hardware": "NVIDIA T4",
  "seed": 42,
  "dataset": "nome_dataset",
  "modello": "descrizione_modello",
  "obiettivo": "cosa si vuole misurare",
  "metriche": {},
  "tempo_totale_sec": null
}
```

## Esperimenti Presenti

### ViT CIFAR-100 (v1) — 5 attivazioni
- `vit_cifar100_{nova,gelu,relu,silu,mish}_20260222.json`

### Scaling ViT (v1) — NOVA vs GELU, 3 scale
- `vit_scaling_{tiny,small,base}_{nova,gelu}_20260222_*.json`

### Scaling ViT v2 (anti-overfitting) — NOVA vs GELU, 3 scale
- `vit_scaling_v2_{tiny,small,base}_{nova,gelu}_20260222_*.json`
- `plot_v2_training_curves.png` — curve di training per scala
- `plot_v2_beta_evolution.png` — evoluzione del parametro β

## Nota

Qualsiasi numero citato nel paper LaTeX (`paper/NOVA.tex`) o nel `README.md` del progetto **deve** avere un file corrispondente in questa cartella che lo giustifichi.
