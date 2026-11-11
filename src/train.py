"""Treina os modelos (MFCC/Wav2Vec2 x 3 regimes de treino) e salva as
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

Regimes de treino:
  - clean: só o canal sem distorção.
  - augmented: os 5 canais de treino juntos (TRAIN_CHANNELS) — 5x mais janelas.
  - augmented_matched: os mesmos 5 canais, mas sorteando só o MESMO número de
    janelas do regime clean. Controle: separa o efeito de "ver canais
    diferentes" do efeito de simplesmente "ter mais dados de treino".

O teste sempre roda em todos os canais, inclusive vonr_loss (VoNR com perda
de pacotes), que nunca entra no treino — mede a robustez a uma degradação
não vista.

MFCC -> RandomForestClassifier (leve, rápido, poucos atributos).
Wav2Vec2 -> LogisticRegression sobre embeddings padronizados (mais rápido
que uma rede neural para este volume de dados em CPU, e um baseline forte
e padrão para embeddings pré-treinados), da camada --layer.

Uso:
    python -m src.train
    python -m src.train --dataset asvspoof
    python -m src.train --dataset asvspoof --save-final   # também salva os modelos finais em models/
    python -m src.train --dataset asvspoof --features wav2vec2 --layer 12 --final-only
        # só os modelos finais de outra camada, para comparação nos testes externos
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.features import W2V_LAYER, load_mfcc, load_wav2vec2

PRED_PATHS = {
    "deepvoice": Path("results/cv_predictions.csv"),
    "asvspoof": Path("results/cv_predictions_asvspoof.csv"),
}
MODELS_DIR = Path("models")

TRAIN_CHANNELS = ("clean", "gsm", "volte", "vonr", "mic")
REGIMES = ("clean", "augmented", "augmented_matched")
RANDOM_SEED = 42
# Colunas de metadado que acompanham cada predição (para análises por locutor,
# por gravação/ligação e por tipo de ataque), quando existem no dataset.
PRED_META_COLS = ("speakers", "source_file", "window_index", "attack_id")


def make_classifier(feature_name: str):
    if feature_name == "mfcc":
        return RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1
        )
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED),
    )


def regime_mask(regime: str, channels: np.ndarray, base_mask: np.ndarray) -> np.ndarray:
    """Janelas de treino de um regime, dentro de base_mask (as janelas permitidas
    no fold/split)."""
    clean_mask = base_mask & (channels == "clean")
    if regime == "clean":
        return clean_mask
    augmented_mask = base_mask & np.isin(channels, TRAIN_CHANNELS)
    if regime == "augmented":
        return augmented_mask
    rng = np.random.default_rng(RANDOM_SEED)
    chosen = rng.choice(np.flatnonzero(augmented_mask), size=int(clean_mask.sum()), replace=False)
    matched = np.zeros_like(base_mask)
    matched[chosen] = True
    return matched


def label_vector(df: pd.DataFrame) -> np.ndarray:
    return (df["label"] == "fake").astype(int).values


def prediction_frame(df: pd.DataFrame, mask: np.ndarray, proba: np.ndarray, **fixed) -> pd.DataFrame:
    sub = df.loc[mask]
    out = pd.DataFrame({**fixed, "channel": sub["channel"].values, "label": label_vector(sub), "proba": proba})
    for col in PRED_META_COLS:
        if col in sub.columns:
            out[col] = sub[col].values
    return out


def run_loso(df: pd.DataFrame, X: np.ndarray, feature_name: str) -> pd.DataFrame:
    y = label_vector(df)
    channels = df["channel"].values
    speaker_lists = df["speakers"].str.split(",")
    speakers = sorted({sp for row in speaker_lists for sp in row})

    fold_results = []
    for held_out in tqdm(speakers, desc=f"{feature_name} · LOSO"):
        is_test_speaker = speaker_lists.apply(lambda row: held_out in row).values

        for regime in REGIMES:
            train_mask = regime_mask(regime, channels, ~is_test_speaker)
            clf = make_classifier(feature_name)
            clf.fit(X[train_mask], y[train_mask])
            proba = clf.predict_proba(X[is_test_speaker])[:, 1]
            fold_results.append(prediction_frame(
                df, is_test_speaker, proba, held_out_speaker=held_out, feature=feature_name, regime=regime,
            ))

    return pd.concat(fold_results, ignore_index=True)


def run_official_split(df: pd.DataFrame, X: np.ndarray, feature_name: str) -> pd.DataFrame:
    """Treina uma vez por regime sobre o split 'train' do ASVspoof; testa nos
    splits 'dev' (locutor novo, ataque conhecido) e 'eval' (locutor E ataque
    novos, o teste de generalização mais forte que o LOSO do DEEP-VOICE)."""
    y = label_vector(df)
    channels = df["channel"].values
    splits = df["split"].values

    fold_results = []
    for regime in tqdm(REGIMES, desc=f"{feature_name} · split oficial"):
        train_mask = regime_mask(regime, channels, splits == "train")
        clf = make_classifier(feature_name)
        clf.fit(X[train_mask], y[train_mask])

        for test_split in ("dev", "eval"):
            test_mask = splits == test_split
            proba = clf.predict_proba(X[test_mask])[:, 1]
            fold_results.append(prediction_frame(
                df, test_mask, proba, split=test_split, feature=feature_name, regime=regime,
            ))

    return pd.concat(fold_results, ignore_index=True)


def final_model_path(feature_name: str, regime: str, dataset: str, layer: int | None = None) -> Path:
    """Caminho do modelo final. Um Wav2Vec2 de outra camada que não W2V_LAYER
    (só para comparação, ver --final-only) ganha a camada no nome."""
    if feature_name == "wav2vec2" and layer not in (None, W2V_LAYER):
        feature_name = f"wav2vec2-l{layer}"
    return MODELS_DIR / f"{feature_name}_{regime}_{dataset}.joblib"


def save_final_models(df: pd.DataFrame, X: np.ndarray, feature_name: str, dataset: str, layer: int | None = None) -> None:
    """Modelos finais (os três regimes) para o teste ao vivo e as
    avaliações externas: no DEEP-VOICE, treinados com todos os locutores; no
    ASVspoof, só com o split de treino (dev/eval continuam nunca vistos)."""
    base_mask = (df["split"].values == "train") if dataset == "asvspoof" else np.ones(len(df), dtype=bool)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for regime in REGIMES:
        train_mask = regime_mask(regime, df["channel"].values, base_mask)
        clf = make_classifier(feature_name)
        clf.fit(X[train_mask], label_vector(df)[train_mask])
        out = final_model_path(feature_name, regime, dataset, layer)
        joblib.dump(clf, out)
        print(f"Modelo final salvo em {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["deepvoice", "asvspoof"], default="deepvoice")
    parser.add_argument("--features", choices=["mfcc", "wav2vec2", "both"], default="both")
    parser.add_argument("--layer", type=int, default=W2V_LAYER, help="camada do Wav2Vec2 (0-12)")
    parser.add_argument("--save-final", action="store_true", help="também treina e salva os modelos finais")
    parser.add_argument(
        "--final-only", action="store_true",
        help="só treina e salva os modelos finais, sem avaliação nem predições (para comparar outra --layer nos testes externos)",
    )
    args = parser.parse_args()

    run = run_official_split if args.dataset == "asvspoof" else run_loso

    parts = []
    for feature_name in ("mfcc", "wav2vec2"):
        if args.features not in (feature_name, "both"):
            continue
        df, X = load_mfcc(args.dataset) if feature_name == "mfcc" else load_wav2vec2(args.dataset, args.layer)
        if not args.final_only:
            parts.append(run(df, X, feature_name))
        if args.save_final or args.final_only:
            save_final_models(df, X, feature_name, args.dataset, args.layer if feature_name == "wav2vec2" else None)
        del X

    if args.final_only:
        return

    results = pd.concat(parts, ignore_index=True)
    out_path = PRED_PATHS[args.dataset]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"Predições salvas em {out_path}: {len(results)} linhas")


if __name__ == "__main__":
    main()
