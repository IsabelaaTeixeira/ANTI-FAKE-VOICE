"""Extração de atributos para cada janela do manifest.

Dois tipos de atributo, para os dois classificadores do projeto:
  - MFCC: descritores espectrais rápidos (mesmo esquema do DATASET-balanced.csv
    original do DEEP-VOICE) — usado no classificador leve (Random Forest).
  - Wav2Vec2: embedding de uma rede neural pré-treinada — usado no classificador
    pesado. Bem mais lento em CPU; processa em lotes e retoma de onde parou.

Uso:
    python -m src.features --kind mfcc [--dataset deepvoice|asvspoof]
    python -m src.features --kind wav2vec2 --dataset asvspoof [--batch-size 16] [--limit 200]
"""
import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

DATASET_PATHS = {
    "deepvoice": {
        "manifest": Path("data/manifest.csv"),
        "mfcc_out": Path("data/features_mfcc.csv"),
        "w2v_meta_out": Path("data/features_wav2vec2_meta.csv"),
        "w2v_array_out": Path("data/features_wav2vec2.npz"),
    },
    "asvspoof": {
        "manifest": Path("data/manifest_asvspoof.csv"),
        "mfcc_out": Path("data/features_mfcc_asvspoof.csv"),
        "w2v_meta_out": Path("data/features_wav2vec2_meta_asvspoof.csv"),
        "w2v_array_out": Path("data/features_wav2vec2_asvspoof.npz"),
    },
}

# Aliases para o dataset padrão (DEEP-VOICE), usados por scripts que só fazem
# sentido nesse dataset: latency_bench.py, live_demo.py, speaker_dependent_eval.py,
# own_voice_ceiling.py, own_voice_pretrain_test.py.
MANIFEST_PATH = DATASET_PATHS["deepvoice"]["manifest"]
MFCC_OUT = DATASET_PATHS["deepvoice"]["mfcc_out"]
W2V_META_OUT = DATASET_PATHS["deepvoice"]["w2v_meta_out"]
W2V_ARRAY_OUT = DATASET_PATHS["deepvoice"]["w2v_array_out"]

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


def run_mfcc(manifest: pd.DataFrame, meta_cols: list[str], mfcc_out: Path) -> None:
    rows = [extract_mfcc(p) for p in tqdm(manifest["filepath"], desc="MFCC")]
    feat = pd.DataFrame(rows, columns=MFCC_COLUMNS)
    out = pd.concat([manifest[meta_cols].reset_index(drop=True), feat], axis=1)
    out.to_csv(mfcc_out, index=False)
    print(f"MFCC salvo em {mfcc_out}: {len(out)} janelas, {len(MFCC_COLUMNS)} atributos.")


def run_wav2vec2(
    manifest: pd.DataFrame, meta_cols: list[str], w2v_array_out: Path, w2v_meta_out: Path,
    batch_size: int, save_every: int, limit: int | None,
) -> None:
    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device).eval()

    rows = manifest if limit is None else manifest.head(limit)
    paths = rows["filepath"].tolist()

    done = 0
    embeddings: list[np.ndarray] = []
    if w2v_array_out.exists() and w2v_meta_out.exists():
        existing_meta = pd.read_csv(w2v_meta_out)
        existing_embeddings = list(np.load(w2v_array_out)["X"])
        n = min(len(existing_meta), len(existing_embeddings), len(paths))
        # só reaproveita se o prefixo bater com o manifest atual (mesmo filepath, na mesma
        # ordem) — evita retomar com embeddings de um manifest antigo/diferente (mesma
        # contagem de linhas não implica mesmo conteúdo, ver fase BB/DD)
        matches = n > 0 and (existing_meta["filepath"].values[:n] == np.array(paths[:n])).all()
        if matches:
            embeddings = existing_embeddings[:n]
            done = n
        else:
            print("Aviso: embeddings existentes não batem com o manifest atual — recalculando do zero.")

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
            np.savez(w2v_array_out, X=np.array(embeddings, dtype=np.float32))
            rows.iloc[: len(embeddings)][meta_cols].to_csv(w2v_meta_out, index=False)
            since_last_save = 0

    print(f"Wav2Vec2 salvo em {w2v_array_out}: {len(embeddings)} janelas, {embeddings[0].shape[0]} dimensões.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["mfcc", "wav2vec2"], required=True)
    parser.add_argument("--dataset", choices=["deepvoice", "asvspoof"], default="deepvoice")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=800)
    parser.add_argument("--limit", type=int, default=None, help="processa só as N primeiras linhas (teste/benchmark)")
    args = parser.parse_args()

    paths = DATASET_PATHS[args.dataset]
    meta_cols = META_COLS + (["split", "attack_id"] if args.dataset == "asvspoof" else [])

    manifest = pd.read_csv(paths["manifest"])
    if args.kind == "mfcc":
        run_mfcc(manifest, meta_cols, paths["mfcc_out"])
    else:
        run_wav2vec2(
            manifest, meta_cols, paths["w2v_array_out"], paths["w2v_meta_out"],
            batch_size=args.batch_size, save_every=args.save_every, limit=args.limit,
        )


if __name__ == "__main__":
    main()
