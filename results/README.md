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

## Nota

Qualsiasi numero citato nel paper LaTeX (`paper/NOVA.tex`) o nel `README.md` del progetto **deve** avere un file corrispondente in questa cartella che lo giustifichi.
