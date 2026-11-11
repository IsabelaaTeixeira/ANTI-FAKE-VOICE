"""Testa cada uma das 13 camadas do Wav2Vec2 como atributo e escolhe a melhor.

Para cada camada, treina o mesmo classificador do projeto (Regressão
Logística, regime augmented) e mede o EER por canal:
  - ASVspoof: treina no split de treino, mede no dev e no eval.
  - DEEP-VOICE: LOSO completo (6 dobras), como em train.py.

A camada escolhida é a de menor EER médio nos canais de treino do split
DEV do ASVspoof — o único conjunto que não é usado para reportar resultado
final. O eval do ASVspoof e o LOSO do DEEP-VOICE aparecem na tabela só para
mostrar se a escolha se confirma em dados que não influenciaram a decisão.
O valor escolhido deve ser copiado para W2V_LAYER em src/features.py.

Uso:
    python -m src.layer_sweep
"""
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.evaluate import eer
from src.features import W2V_N_LAYERS, load_wav2vec2
from src.train import TRAIN_CHANNELS, label_vector, make_classifier, regime_mask

OUT_PATH = Path("results/layer_sweep.csv")


def per_channel_eer(test_df: pd.DataFrame, proba: np.ndarray, **fixed) -> list[dict]:
    y = label_vector(test_df)
    channels = test_df["channel"].values
    return [
        {**fixed, "channel": ch, "eer": eer(y[channels == ch], proba[channels == ch])}
        for ch in np.unique(channels)
    ]


def sweep_asvspoof() -> list[dict]:
    rows = []
    for layer in tqdm(range(W2V_N_LAYERS), desc="ASVspoof · camadas"):
        df, X = load_wav2vec2("asvspoof", layer)
        splits = df["split"].values
        train_mask = regime_mask("augmented", df["channel"].values, splits == "train")
        clf = make_classifier("wav2vec2").fit(X[train_mask], label_vector(df)[train_mask])
        for split in ("dev", "eval"):
            mask = splits == split
            rows += per_channel_eer(df[mask], clf.predict_proba(X[mask])[:, 1], dataset="asvspoof", split=split, layer=layer)
    return rows


def sweep_deepvoice() -> list[dict]:
    rows = []
    for layer in tqdm(range(W2V_N_LAYERS), desc="DEEP-VOICE · camadas"):
        df, X = load_wav2vec2("deepvoice", layer)
        y = label_vector(df)
        speaker_lists = df["speakers"].str.split(",")
        speakers = sorted({sp for row in speaker_lists for sp in row})
        probas, test_parts = [], []
        for held_out in speakers:
            is_test = speaker_lists.apply(lambda row: held_out in row).values
            train_mask = regime_mask("augmented", df["channel"].values, ~is_test)
            clf = make_classifier("wav2vec2").fit(X[train_mask], y[train_mask])
            probas.append(clf.predict_proba(X[is_test])[:, 1])
            test_parts.append(df[is_test])
        rows += per_channel_eer(pd.concat(test_parts), np.concatenate(probas), dataset="deepvoice", split="loso", layer=layer)
    return rows


def main() -> None:
    table = pd.DataFrame(sweep_asvspoof() + sweep_deepvoice())
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_PATH, index=False)

    summary = (
        table[table["channel"].isin(TRAIN_CHANNELS)]
        .groupby(["dataset", "split", "layer"])["eer"].mean()
        .unstack(["dataset", "split"])
    )
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print("EER médio nos canais de treino, por camada:")
    print(summary.to_string())
    best = int(summary[("asvspoof", "dev")].idxmin())
    print(f"\nCamada escolhida (menor EER no dev do ASVspoof): {best} — copie para W2V_LAYER em src/features.py")
    print(f"Tabela salva em {OUT_PATH}")


if __name__ == "__main__":
    main()
