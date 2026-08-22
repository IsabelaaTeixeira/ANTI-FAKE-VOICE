"""Treina os 4 modelos (MFCC/Wav2Vec2 x limpo/augmentado) e salva as
predições de teste para a tabela de avaliação da etapa 5.

Duas estratégias de avaliação, escolhidas por --dataset:

  - deepvoice (padrão): validação cruzada Leave-One-Speaker-Out (LOSO). Para
    cada um dos 6 locutores, ele é totalmente excluído do treino (nenhuma
    janela onde ele aparece, seja como voz real ou como origem/destino de
    uma conversão, entra no fit) e usado só para teste.
  - asvspoof: o próprio corpus já separa locutores por split (treino não
    repete ninguém de validação/teste) e o split de teste já usa tipos de
    ataque nunca vistos no treino/validação. Por isso aqui não se refaz o
    LOSO manual: cada regime é treinado uma vez só sobre o split de treino,
    e testado nos splits de validação ("dev", locutor novo mas ataque
    conhecido) e teste ("eval", locutor E ataque novos).

Em ambos os casos, o regime "augmented" treina com os 4 canais juntos
(clean+gsm+volte+vonr); o "clean" treina só com o canal sem distorção. O
teste sempre roda nos 4 canais, para alimentar a tabela canal x modelo.

MFCC -> RandomForestClassifier (leve, rápido, poucos atributos).
Wav2Vec2 -> LogisticRegression sobre embeddings padronizados (mais rápido
que uma rede neural para este volume de dados em CPU, e um baseline forte
e padrão para embeddings pré-treinados).

Uso:
    python -m src.train
    python -m src.train --dataset asvspoof
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

DATASET_PATHS = {
    "deepvoice": {
        "mfcc": Path("data/features_mfcc.csv"),
        "w2v_array": Path("data/features_wav2vec2.npz"),
        "w2v_meta": Path("data/features_wav2vec2_meta.csv"),
        "pred_out": Path("results/cv_predictions.csv"),
    },
    "asvspoof": {
        "mfcc": Path("data/features_mfcc_asvspoof.csv"),
        "w2v_array": Path("data/features_wav2vec2_asvspoof.npz"),
        "w2v_meta": Path("data/features_wav2vec2_meta_asvspoof.csv"),
        "pred_out": Path("results/cv_predictions_asvspoof.csv"),
    },
}

META_COLS_BASE = ("filepath", "speakers", "label", "channel")
REGIMES = ("clean", "augmented")


def load_mfcc(path: Path, meta_cols: tuple[str, ...]) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c not in meta_cols]
    return df, df[feat_cols].values


def load_wav2vec2(array_path: Path, meta_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    meta = pd.read_csv(meta_path)
    X = np.load(array_path)["X"]
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


def run_official_split(df: pd.DataFrame, X: np.ndarray, feature_name: str) -> pd.DataFrame:
    """Treina uma vez por regime sobre o split 'train' do ASVspoof; testa nos
    splits 'dev' (locutor novo, ataque conhecido) e 'eval' (locutor E ataque
    novos, o teste de generalização mais forte que o LOSO do DEEP-VOICE)."""
    y = (df["label"] == "fake").astype(int).values
    channels = df["channel"].values
    splits = df["split"].values
    is_train_split = splits == "train"

    fold_results = []
    for regime in tqdm(REGIMES, desc=f"{feature_name} · split oficial"):
        channel_ok = (channels == "clean") if regime == "clean" else np.ones(len(df), dtype=bool)
        train_mask = is_train_split & channel_ok

        clf = make_classifier(feature_name)
        clf.fit(X[train_mask], y[train_mask])

        for test_split in ("dev", "eval"):
            test_mask = splits == test_split
            proba = clf.predict_proba(X[test_mask])[:, 1]
            fold_results.append(pd.DataFrame({
                "split": test_split,
                "feature": feature_name,
                "regime": regime,
                "channel": channels[test_mask],
                "label": y[test_mask],
                "proba": proba,
            }))

    return pd.concat(fold_results, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["deepvoice", "asvspoof"], default="deepvoice")
    parser.add_argument("--features", choices=["mfcc", "wav2vec2", "both"], default="both")
    args = parser.parse_args()

    paths = DATASET_PATHS[args.dataset]
    meta_cols = META_COLS_BASE + (("split", "attack_id") if args.dataset == "asvspoof" else ())
    run = run_official_split if args.dataset == "asvspoof" else run_loso

    parts = []
    if args.features in ("mfcc", "both"):
        mfcc_df, X_mfcc = load_mfcc(paths["mfcc"], meta_cols)
        parts.append(run(mfcc_df, X_mfcc, "mfcc"))

    if args.features in ("wav2vec2", "both"):
        w2v_meta, X_w2v = load_wav2vec2(paths["w2v_array"], paths["w2v_meta"])
        parts.append(run(w2v_meta, X_w2v, "wav2vec2"))

    results = pd.concat(parts, ignore_index=True)
    paths["pred_out"].parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(paths["pred_out"], index=False)
    print(f"Predições salvas em {paths['pred_out']}: {len(results)} linhas")


if __name__ == "__main__":
    main()
