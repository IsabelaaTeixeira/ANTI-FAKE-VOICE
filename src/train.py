"""Treina os 4 modelos (MFCC/Wav2Vec2 × limpo/augmentado) com validação
cruzada Leave-One-Speaker-Out (LOSO) e salva as predições fora-da-dobra
para a tabela de avaliação da etapa 5.

Para cada um dos 6 locutores, ele é totalmente excluído do treino (nenhuma
janela onde ele aparece, seja como voz real ou como origem/destino de uma
conversão, entra no fit) e usado só para teste. O regime "augmented" treina
com os 4 canais juntos (clean+gsm+volte+vonr); o "clean" treina só com o
canal sem distorção. O teste sempre roda nos 4 canais, para alimentar a
tabela canal × modelo.

MFCC -> RandomForestClassifier (leve, rápido, poucos atributos).
Wav2Vec2 -> LogisticRegression sobre embeddings padronizados (mais rápido
que uma rede neural para este volume de dados em CPU, e um baseline forte
e padrão para embeddings pré-treinados).

Uso:
    python -m src.train
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

MFCC_PATH = Path("data/features_mfcc.csv")
W2V_ARRAY_PATH = Path("data/features_wav2vec2.npz")
W2V_META_PATH = Path("data/features_wav2vec2_meta.csv")
PRED_OUT = Path("results/cv_predictions.csv")

META_COLS = ("filepath", "speakers", "label", "channel")
REGIMES = ("clean", "augmented")


def load_mfcc() -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(MFCC_PATH)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    return df, df[feat_cols].values


def load_wav2vec2() -> tuple[pd.DataFrame, np.ndarray]:
    meta = pd.read_csv(W2V_META_PATH)
    X = np.load(W2V_ARRAY_PATH)["X"]
    return meta, X


def make_classifier(feature_name: str):
    if feature_name == "mfcc":
        return RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
        )
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    )


def run_loso(df: pd.DataFrame, X: np.ndarray, feature_name: str) -> pd.DataFrame:
    y = (df["label"] == "fake").astype(int).values
    channels = df["channel"].values
    speaker_lists = df["speakers"].str.split(",")
    speakers = sorted({sp for row in speaker_lists for sp in row})

    fold_results = []
    for held_out in tqdm(speakers, desc=f"{feature_name} · LOSO"):
        is_test_speaker = speaker_lists.apply(lambda row: held_out in row).values

        for regime in REGIMES:
            channel_ok = (channels == "clean") if regime == "clean" else np.ones(len(df), dtype=bool)
            train_mask = (~is_test_speaker) & channel_ok
            test_mask = is_test_speaker

            clf = make_classifier(feature_name)
            clf.fit(X[train_mask], y[train_mask])
            proba = clf.predict_proba(X[test_mask])[:, 1]

            fold_results.append(pd.DataFrame({
                "held_out_speaker": held_out,
                "feature": feature_name,
                "regime": regime,
                "channel": channels[test_mask],
                "label": y[test_mask],
                "proba": proba,
            }))

    return pd.concat(fold_results, ignore_index=True)


def main() -> None:
    mfcc_df, X_mfcc = load_mfcc()
    res_mfcc = run_loso(mfcc_df, X_mfcc, "mfcc")

    w2v_meta, X_w2v = load_wav2vec2()
    res_w2v = run_loso(w2v_meta, X_w2v, "wav2vec2")

    results = pd.concat([res_mfcc, res_w2v], ignore_index=True)
    PRED_OUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(PRED_OUT, index=False)
    print(f"Predições (fora da dobra) salvas em {PRED_OUT}: {len(results)} linhas")


if __name__ == "__main__":
    main()
