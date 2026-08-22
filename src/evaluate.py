"""Calcula acurácia, F1 e EER por canal x modelo (e, no ASVspoof, por split)
a partir das predições de src/train.py, montando a tabela final da etapa 5.

EER (Equal Error Rate): o ponto na curva ROC onde a taxa de falso positivo
(marcar voz real como clonada) se iguala à taxa de falso negativo (deixar
passar uma voz clonada) — métrica padrão em detecção de fraude/biometria,
mais robusta a desbalanceamento de classe do que a acurácia crua.

Uso:
    python -m src.evaluate
    python -m src.evaluate --dataset asvspoof
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_curve

DATASET_PATHS = {
    "deepvoice": {
        "pred": Path("results/cv_predictions.csv"),
        "table_out": Path("results/channel_model_table.csv"),
    },
    "asvspoof": {
        "pred": Path("results/cv_predictions_asvspoof.csv"),
        "table_out": Path("results/channel_model_table_asvspoof.csv"),
    },
}


def eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["deepvoice", "asvspoof"], default="deepvoice")
    args = parser.parse_args()
    paths = DATASET_PATHS[args.dataset]

    df = pd.read_csv(paths["pred"])
    df["pred"] = (df["proba"] >= 0.5).astype(int)

    group_cols = ["feature", "regime", "channel"]
    if args.dataset == "asvspoof":
        group_cols = ["split"] + group_cols

    rows = []
    for key, g in df.groupby(group_cols):
        row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        row.update({
            "n": len(g),
            "accuracy": (g["pred"] == g["label"]).mean(),
            "f1": f1_score(g["label"], g["pred"], pos_label=1, zero_division=0),
            "eer": eer(g["label"].values, g["proba"].values),
        })
        rows.append(row)

    table = pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)
    paths["table_out"].parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(paths["table_out"], index=False)

    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(table.to_string(index=False))
    print(f"\nTabela salva em {paths['table_out']}")


if __name__ == "__main__":
    main()
