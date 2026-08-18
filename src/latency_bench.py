"""Mede a latência de inferência por janela de 1s de cada arquitetura, em
CPU (perfil de hardware MEC leve, sem GPU) — etapa 6.

Cronometra o pipeline completo como aconteceria numa ligação ao vivo: áudio
de 1s já em memória (sem custo de ler arquivo do disco) -> extração de
atributos -> predição do classificador. Repete várias vezes por arquitetura
para tirar média, mediana e p95 (o pior caso importa mais que a média num
sistema em tempo real).

Uso:
    python -m src.latency_bench
"""
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from src.features import META_COLS, MFCC_OUT, SR, W2V_ARRAY_OUT, W2V_META_OUT, extract_mfcc_from_array

MANIFEST_PATH = Path("data/manifest.csv")
LATENCY_OUT = Path("results/latency.csv")

N_BENCH_MFCC = 300
N_BENCH_W2V = 60

WINDOW_MS = 1000          # duração da própria janela: o processamento tem que ser mais rápido que isso
LIVE_CALL_BUDGET_MS = 150  # ITU-T G.114: atraso boca-ouvido aceitável numa ligação conversacional


def bench_mfcc(sample_paths: list[str]) -> np.ndarray:
    mfcc_df = pd.read_csv(MFCC_OUT)
    feat_cols = [c for c in mfcc_df.columns if c not in META_COLS]
    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=1)
    clf.fit(mfcc_df[feat_cols].values, (mfcc_df["label"] == "fake").astype(int).values)

    audios = [librosa.load(p, sr=SR, mono=True)[0] for p in sample_paths]

    times = []
    for y in audios:
        t0 = time.perf_counter()
        feat = extract_mfcc_from_array(y)
        clf.predict_proba([feat])
        times.append((time.perf_counter() - t0) * 1000)
    return np.array(times)


def bench_wav2vec2(sample_paths: list[str]) -> np.ndarray:
    w2v_meta = pd.read_csv(W2V_META_OUT)
    X = np.load(W2V_ARRAY_OUT)["X"]
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    )
    clf.fit(X, (w2v_meta["label"] == "fake").astype(int).values)

    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").eval()

    audios = [librosa.load(p, sr=SR, mono=True)[0] for p in sample_paths]

    times = []
    for y in audios:
        t0 = time.perf_counter()
        inputs = processor([y], sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            out = model(inputs.input_values)
        embedding = out.last_hidden_state.mean(dim=1).numpy()
        clf.predict_proba(embedding)
        times.append((time.perf_counter() - t0) * 1000)
    return np.array(times)


def summarize(name: str, times: np.ndarray) -> dict:
    return {
        "arquitetura": name,
        "n": len(times),
        "media_ms": times.mean(),
        "mediana_ms": np.median(times),
        "p95_ms": np.percentile(times, 95),
        "max_ms": times.max(),
        "cabe_na_janela_1s": bool(np.percentile(times, 95) < WINDOW_MS),
        "cabe_no_orcamento_call_150ms": bool(np.percentile(times, 95) < LIVE_CALL_BUDGET_MS),
    }


def main() -> None:
    manifest = pd.read_csv(MANIFEST_PATH)
    clean_paths = manifest.loc[manifest["channel"] == "clean", "filepath"].tolist()

    print(f"Benchmark MFCC ({N_BENCH_MFCC} janelas)...")
    mfcc_times = bench_mfcc(clean_paths[:N_BENCH_MFCC])

    print(f"Benchmark Wav2Vec2 ({N_BENCH_W2V} janelas)...")
    w2v_times = bench_wav2vec2(clean_paths[:N_BENCH_W2V])

    table = pd.DataFrame([
        summarize("MFCC + Random Forest", mfcc_times),
        summarize("Wav2Vec2 + Regressão Logística", w2v_times),
    ])

    LATENCY_OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(LATENCY_OUT, index=False)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    print("\n" + table.to_string(index=False))
    print(f"\nTabela salva em {LATENCY_OUT}")


if __name__ == "__main__":
    main()
