"""Avalia o modelo final (models/mfcc_rf_final.joblib, treinado com TODOS
os dados) nas próprias vozes que estavam no treino — ideia 1 da usuária.

Diferente do LOSO (etapas 5-6), aqui o modelo JÁ viu a voz da pessoa
durante o treino. Serve como "teto" de desempenho: ajuda a separar quanto
da queda vista no LOSO é por causa do canal (GSM/VoLTE/VoNR) e quanto é
por não conhecer a pessoa.

AVISO METODOLÓGICO: isso é desempenho sobre dados que estavam no conjunto
de treino, não um teste de generalização — os números saem mais altos que
o LOSO por natureza. Serve como referência de teto, não para comparar
"qual modelo é melhor" com a tabela da etapa 5.

Uso:
    python -m src.speaker_dependent_eval
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score, roc_curve

from src.features import META_COLS, MFCC_OUT

MODEL_PATH = "models/mfcc_rf_final.joblib"
SPEAKERS = ["taylor", "biden", "ryan"]


def eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2)


def main() -> None:
    clf = joblib.load(MODEL_PATH)
    df = pd.read_csv(MFCC_OUT)
    feat_cols = [c for c in df.columns if c not in META_COLS]

    rows = []
    for speaker in SPEAKERS:
        sub_speaker = df[df["speakers"].str.contains(speaker)]
        for channel, sub in sub_speaker.groupby("channel"):
            X = sub[feat_cols].values
            y = (sub["label"] == "fake").astype(int).values
            proba = clf.predict_proba(X)[:, 1]
            pred = (proba >= 0.5).astype(int)
            rows.append({
                "speaker": speaker,
                "channel": channel,
                "n": len(sub),
                "accuracy": (pred == y).mean(),
                "f1": f1_score(y, pred, pos_label=1, zero_division=0),
                "eer": eer(y, proba),
            })

    table = pd.DataFrame(rows).sort_values(["speaker", "channel"]).reset_index(drop=True)
    out_path = "results/speaker_dependent_table.csv"
    table.to_csv(out_path, index=False)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(table.to_string(index=False))
    print(f"\nTabela salva em {out_path}")


if __name__ == "__main__":
    main()
