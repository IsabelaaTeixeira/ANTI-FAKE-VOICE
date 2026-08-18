"""Extração de atributos para cada janela do manifest.

Dois tipos de atributo, para os dois classificadores do projeto:
  - MFCC: descritores espectrais rápidos (mesmo esquema do DATASET-balanced.csv
    original do DEEP-VOICE) — usado no classificador leve (Random Forest).
  - Wav2Vec2: embedding de uma rede neural pré-treinada — usado no classificador
    pesado. Bem mais lento em CPU; processa em lotes e retoma de onde parou.

Uso:
    python -m src.features --kind mfcc
    python -m src.features --kind wav2vec2 [--batch-size 16] [--limit 200]
"""
import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

MANIFEST_PATH = Path("data/manifest.csv")
MFCC_OUT = Path("data/features_mfcc.csv")
W2V_META_OUT = Path("data/features_wav2vec2_meta.csv")
W2V_ARRAY_OUT = Path("data/features_wav2vec2.npz")

SR = 16000
N_MFCC = 20
META_COLS = ["filepath", "speakers", "label", "channel"]

MFCC_COLUMNS = [
    "chroma_stft", "rms", "spectral_centroid", "spectral_bandwidth",
    "rolloff", "zero_crossing_rate",
] + [f"mfcc{i}" for i in range(1, N_MFCC + 1)]


def extract_mfcc_from_array(y: np.ndarray) -> list[float]:
    row = [
        librosa.feature.chroma_stft(y=y, sr=SR).mean(),
        librosa.feature.rms(y=y).mean(),
        librosa.feature.spectral_centroid(y=y, sr=SR).mean(),
        librosa.feature.spectral_bandwidth(y=y, sr=SR).mean(),
        librosa.feature.spectral_rolloff(y=y, sr=SR).mean(),
        librosa.feature.zero_crossing_rate(y=y).mean(),
    ]
    row.extend(librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC).mean(axis=1).tolist())
    return row


def extract_mfcc(path: str) -> list[float]:
    y, _ = librosa.load(path, sr=SR, mono=True)
    return extract_mfcc_from_array(y)


def run_mfcc(manifest: pd.DataFrame) -> None:
    rows = [extract_mfcc(p) for p in tqdm(manifest["filepath"], desc="MFCC")]
    feat = pd.DataFrame(rows, columns=MFCC_COLUMNS)
    out = pd.concat([manifest[META_COLS].reset_index(drop=True), feat], axis=1)
    out.to_csv(MFCC_OUT, index=False)
    print(f"MFCC salvo em {MFCC_OUT}: {len(out)} janelas, {len(MFCC_COLUMNS)} atributos.")


def run_wav2vec2(manifest: pd.DataFrame, batch_size: int, save_every: int, limit: int | None) -> None:
    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device).eval()

    rows = manifest if limit is None else manifest.head(limit)
    paths = rows["filepath"].tolist()

    done = 0
    embeddings: list[np.ndarray] = []
    if W2V_ARRAY_OUT.exists() and W2V_META_OUT.exists():
        embeddings = list(np.load(W2V_ARRAY_OUT)["X"])
        done = len(embeddings)

    since_last_save = 0
    for i in tqdm(range(done, len(paths), batch_size), desc="Wav2Vec2", initial=done // batch_size):
        batch_paths = paths[i:i + batch_size]
        audios = [librosa.load(p, sr=SR, mono=True)[0] for p in batch_paths]
        inputs = processor(audios, sampling_rate=SR, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = model(inputs.input_values.to(device))
        pooled = out.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.extend(pooled)
        since_last_save += len(batch_paths)

        if since_last_save >= save_every or i + batch_size >= len(paths):
            np.savez(W2V_ARRAY_OUT, X=np.array(embeddings, dtype=np.float32))
            rows.iloc[: len(embeddings)][META_COLS].to_csv(W2V_META_OUT, index=False)
            since_last_save = 0

    print(f"Wav2Vec2 salvo em {W2V_ARRAY_OUT}: {len(embeddings)} janelas, {embeddings[0].shape[0]} dimensões.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["mfcc", "wav2vec2"], required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=800)
    parser.add_argument("--limit", type=int, default=None, help="processa só as N primeiras linhas (teste/benchmark)")
    args = parser.parse_args()

    manifest = pd.read_csv(MANIFEST_PATH)
    if args.kind == "mfcc":
        run_mfcc(manifest)
    else:
        run_wav2vec2(manifest, batch_size=args.batch_size, save_every=args.save_every, limit=args.limit)


if __name__ == "__main__":
    main()
