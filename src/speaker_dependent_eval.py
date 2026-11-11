"""Locutor conhecido, trecho novo: quanto o detector melhora quando já ouviu
a voz da pessoa durante o treino — o contraponto do LOSO (locutor nunca visto).

Cada gravação é dividida no tempo: as primeiras 70% janelas vão para o
treino, as 5% seguintes são descartadas (separação, para nenhuma janela de
teste ficar colada numa de treino) e as 25% finais vão para o teste. Assim
o modelo conhece todos os locutores, mas é avaliado em trechos de fala que
nunca viu. (Avaliar nas próprias janelas de treino, como numa versão
anterior deste script, mede só memorização: uma Random Forest acerta 100%
do que decorou.)

Treina no regime augmented (canais de treino de train.py) e testa em todos
os canais, para os dois atributos.

Uso:
    python -m src.speaker_dependent_eval
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from src.evaluate import bootstrap_eer_ci, eer
from src.features import load_mfcc, load_wav2vec2
from src.train import label_vector, make_classifier, regime_mask

OUT_PATH = Path("results/speaker_dependent_table.csv")
TRAIN_FRACTION = 0.70
GAP_FRACTION = 0.05


def temporal_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    n_windows = df.groupby("source_file")["window_index"].transform("max") + 1
    position = df["window_index"] / n_windows
    is_train = (position < TRAIN_FRACTION).values
    is_test = (position >= TRAIN_FRACTION + GAP_FRACTION).values
    return is_train, is_test


def evaluate_feature(feature_name: str) -> list[dict]:
    df, X = load_mfcc("deepvoice") if feature_name == "mfcc" else load_wav2vec2("deepvoice")
    y = label_vector(df)
    is_train, is_test = temporal_split(df)
    train_mask = regime_mask("augmented", df["channel"].values, is_train)
    clf = make_classifier(feature_name).fit(X[train_mask], y[train_mask])

    test = df[is_test].assign(label=y[is_test], proba=clf.predict_proba(X[is_test])[:, 1])
    speakers = sorted({sp for row in df["speakers"].str.split(",") for sp in row})
    rows = []
    for channel, g in test.groupby("channel"):
        groups = {"todos": g, **{sp: g[g["speakers"].str.split(",").apply(lambda r, sp=sp: sp in r)] for sp in speakers}}
        for speaker, sub in groups.items():
            y_sub, proba = sub["label"].values, sub["proba"].values
            row = {
                "feature": feature_name, "speaker": speaker, "channel": channel, "n": len(sub),
                "balanced_accuracy": balanced_accuracy_score(y_sub, (proba >= 0.5).astype(int)),
                "eer": eer(y_sub, proba),
            }
            if speaker == "todos":
                row["eer_ci_low"], row["eer_ci_high"] = bootstrap_eer_ci(y_sub, proba, sub["source_file"].values)
            rows.append(row)
    return rows


def main() -> None:
    table = pd.DataFrame(evaluate_feature("mfcc") + evaluate_feature("wav2vec2"))
    table = table.sort_values(["feature", "speaker", "channel"]).reset_index(drop=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_PATH, index=False)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(table[table["speaker"] == "todos"].to_string(index=False))
    print(f"\nTabela salva em {OUT_PATH}")


if __name__ == "__main__":
    main()
