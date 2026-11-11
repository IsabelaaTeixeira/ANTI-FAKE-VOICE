"""Mede a latência de inferência por janela de 1s de cada arquitetura, em
CPU (perfil de hardware MEC leve, sem GPU) — etapa 6.

Cronometra o pipeline completo como aconteceria numa ligação ao vivo: áudio
de 1s já em memória (sem custo de ler arquivo do disco) -> extração de
atributos -> predição do classificador. Repete várias vezes por arquitetura
para tirar média, mediana e p95 (o pior caso importa mais que a média num
sistema em tempo real).

Arquiteturas:
  - MFCC + Random Forest;
  - Wav2Vec2 completo (12 camadas) + Regressão Logística, usando a camada 12;
  - Wav2Vec2 cortado na camada W2V_LAYER: as camadas acima dela não são
    calculadas (a saída da camada mantida é idêntica à da rede completa);
  - o mesmo, quantizado em int8 (pesos das camadas lineares guardados como
    inteiros de 8 bits, mais rápidos em CPU). A quantização muda levemente
    os números, então também se mede o EER antes/depois no split dev do
    ASVspoof (todas as janelas do canal limpo) — results/latency_quality.csv.

Dois orçamentos de tempo:
  - 1000ms (duração da janela): o detector roda em PARALELO à ligação, numa
    cópia do áudio, e só precisa acompanhar o ritmo da fala — não atrasa a
    voz de ninguém. É o requisito de uma implantação como monitor.
  - 150ms (ITU-T G.114, atraso boca-ouvido): só se aplicaria se o detector
    ficasse NO CAMINHO da voz (a ligação esperando o veredito de cada janela).

Uso:
    python -m src.latency_bench
"""
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch

from src.evaluate import eer
from src.features import (
    MANIFEST_PATH, SR, W2V_LAYER, extract_mfcc_from_array, load_mfcc, load_wav2vec2, load_wav2vec2_model,
)
from src.train import label_vector, make_classifier, regime_mask

LATENCY_OUT = Path("results/latency.csv")
QUALITY_OUT = Path("results/latency_quality.csv")

N_BENCH_MFCC = 300
N_BENCH_W2V = 200
N_FIT = 20000  # janelas usadas para ajustar o classificador do benchmark (a latência não depende disso)
TORCH_THREADS = 4  # núcleos físicos deste notebook

WINDOW_MS = 1000          # duração da própria janela: o processamento tem que ser mais rápido que isso
LIVE_CALL_BUDGET_MS = 150  # ITU-T G.114: atraso boca-ouvido aceitável numa ligação conversacional


def _subsample(n: int, k: int) -> np.ndarray:
    return np.random.default_rng(42).choice(n, size=min(n, k), replace=False)


def bench_mfcc(audios: list[np.ndarray]) -> np.ndarray:
    df, X = load_mfcc("deepvoice")
    idx = _subsample(len(df), N_FIT)
    clf = make_classifier("mfcc").set_params(n_jobs=1).fit(X[idx], label_vector(df)[idx])

    times = []
    for y in audios:
        t0 = time.perf_counter()
        clf.predict_proba([extract_mfcc_from_array(y)])
        times.append((time.perf_counter() - t0) * 1000)
    return np.array(times)


def wav2vec2_variant(layer: int, int8: bool):
    processor, model = load_wav2vec2_model(n_layers=layer)
    if int8:
        model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return processor, model


def embed(audios: list[np.ndarray], processor, model) -> np.ndarray:
    inputs = processor(audios, sampling_rate=SR, return_tensors="pt", padding=True)
    with torch.inference_mode():
        return model(inputs.input_values).last_hidden_state.mean(dim=1).numpy()


def bench_wav2vec2(audios: list[np.ndarray], layer: int, int8: bool) -> np.ndarray:
    df, X = load_wav2vec2("deepvoice", layer)
    idx = _subsample(len(df), N_FIT)
    clf = make_classifier("wav2vec2").fit(X[idx], label_vector(df)[idx])
    processor, model = wav2vec2_variant(layer, int8)

    embed(audios[:3], processor, model)  # aquecimento
    times = []
    for y in audios:
        t0 = time.perf_counter()
        clf.predict_proba(embed([y], processor, model))
        times.append((time.perf_counter() - t0) * 1000)
    return np.array(times)


def int8_quality(layer: int) -> pd.DataFrame:
    """EER no canal limpo do dev do ASVspoof com o mesmo classificador (o do
    regime augmented, treinado nos atributos float32 do split de treino),
    extraindo os atributos de teste com a rede float32 e com a rede int8."""
    df, X = load_wav2vec2("asvspoof", layer)
    train = regime_mask("augmented", df["channel"].values, (df["split"] == "train").values)
    clf = make_classifier("wav2vec2").fit(X[train], label_vector(df)[train])
    test = df[(df["split"] == "dev") & (df["channel"] == "clean")]
    audios = [librosa.load(p, sr=SR, mono=True)[0] for p in test["filepath"]]
    y = label_vector(test)

    rows = []
    for int8 in (False, True):
        processor, model = wav2vec2_variant(layer, int8)
        emb = np.concatenate([embed(audios[i:i + 16], processor, model) for i in range(0, len(audios), 16)])
        rows.append({
            "camada": layer, "precisao": "int8" if int8 else "float32", "n": len(test),
            "eer_dev_clean": eer(y, clf.predict_proba(emb)[:, 1]),
        })
    return pd.DataFrame(rows)


def summarize(name: str, times: np.ndarray) -> dict:
    p95 = np.percentile(times, 95)
    return {
        "arquitetura": name,
        "n": len(times),
        "media_ms": times.mean(),
        "mediana_ms": np.median(times),
        "p95_ms": p95,
        "max_ms": times.max(),
        "fator_tempo_real_p95": p95 / WINDOW_MS,
        "cabe_em_paralelo_1s": bool(p95 < WINDOW_MS),
        "cabe_no_caminho_da_voz_150ms": bool(p95 < LIVE_CALL_BUDGET_MS),
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-quality", action="store_true", help="só refaz a comparação de qualidade float32 x int8")
    args = parser.parse_args()
    torch.set_num_threads(TORCH_THREADS)
    if args.only_quality:
        run_quality()
        return
    manifest = pd.read_csv(MANIFEST_PATH)
    clean_paths = manifest.loc[manifest["channel"] == "clean", "filepath"]
    sample = clean_paths.iloc[_subsample(len(clean_paths), max(N_BENCH_MFCC, N_BENCH_W2V))].tolist()
    audios = [librosa.load(p, sr=SR, mono=True)[0] for p in sample]

    rows = []
    print(f"Benchmark MFCC ({N_BENCH_MFCC} janelas)...")
    rows.append(summarize("MFCC + Random Forest", bench_mfcc(audios[:N_BENCH_MFCC])))

    variants = [("Wav2Vec2 completo (camada 12) + Regressão Logística", 12, False)]
    if W2V_LAYER != 12:
        variants.append((f"Wav2Vec2 cortado na camada {W2V_LAYER} + Regressão Logística", W2V_LAYER, False))
    variants.append((f"Wav2Vec2 cortado na camada {W2V_LAYER}, int8 + Regressão Logística", W2V_LAYER, True))
    for name, layer, int8 in variants:
        print(f"Benchmark {name} ({N_BENCH_W2V} janelas)...")
        rows.append(summarize(name, bench_wav2vec2(audios[:N_BENCH_W2V], layer, int8)))

    table = pd.DataFrame(rows)
    LATENCY_OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(LATENCY_OUT, index=False)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")
    pd.set_option("display.width", 250)
    print("\n" + table.to_string(index=False))
    print(f"\nTabela salva em {LATENCY_OUT}")
    run_quality()


def run_quality() -> None:
    print(f"\nQualidade float32 vs int8 (camada {W2V_LAYER}, dev do ASVspoof, canal limpo)...")
    quality = int8_quality(W2V_LAYER)
    quality.to_csv(QUALITY_OUT, index=False)
    print(quality.to_string(index=False))
    print(f"Tabela salva em {QUALITY_OUT}")


if __name__ == "__main__":
    main()
