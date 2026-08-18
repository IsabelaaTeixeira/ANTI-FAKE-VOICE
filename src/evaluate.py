"""Calcula acurácia, F1 e EER por canal × modelo a partir das predições
fora-da-dobra de src/train.py, montando a tabela final da etapa 5.

EER (Equal Error Rate): o ponto na curva ROC onde a taxa de falso positivo
(marcar voz real como clonada) se iguala à taxa de falso negativo (deixar
passar uma voz clonada) — métrica padrão em detecção de fraude/biometria,
mais robusta a desbalanceamento de classe do que a acurácia crua.

Uso:
    python -m src.evaluate
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_curve

PRED_PATH = Path("results/cv_predictions.csv")
TABLE_OUT = Path("results/channel_model_table.csv")


def eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2)


def main() -> None:
    df = pd.read_csv(PRED_PATH)
    df["pred"] = (df["proba"] >= 0.5).astype(int)

    rows = []
    for (feature, regime, channel), g in df.groupby(["feature", "regime", "channel"]):
        rows.append({
            "feature": feature,
            "regime": regime,
            "channel": channel,
            "n": len(g),
            "accuracy": (g["pred"] == g["label"]).mean(),
            "f1": f1_score(g["label"], g["pred"], pos_label=1, zero_division=0),
            "eer": eer(g["label"].values, g["proba"].values),
        })

    table = pd.DataFrame(rows).sort_values(["feature", "regime", "channel"]).reset_index(drop=True)
    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_OUT, index=False)

    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(table.to_string(index=False))
    print(f"\nTabela salva em {TABLE_OUT}")


if __name__ == "__main__":
    main()
