"""Extração de atributos para cada janela do manifest.

Dois tipos de atributo, para os dois classificadores do projeto:
  - MFCC: descritores espectrais rápidos (mesmo esquema do DATASET-balanced.csv
    original do DEEP-VOICE) — usado no classificador leve (Random Forest).
  - Wav2Vec2: embedding de uma rede neural pré-treinada — usado no classificador
    pesado. Salva a média no tempo de TODAS as 13 camadas da rede (0 = saída do
    bloco convolucional, 1..12 = camadas do transformer), não só da última: a
    literatura de detecção de spoofing mostra que camadas intermediárias costumam
    carregar mais pistas acústicas de síntese, e a última, mais conteúdo
    linguístico. A camada usada pelos classificadores (W2V_LAYER) é escolhida
    por src/layer_sweep.py. Bem mais lento em CPU; processa em lotes e retoma de
    onde parou.

Uso:
    python -m src.features --kind mfcc [--dataset deepvoice|asvspoof|pt]
    python -m src.features --kind wav2vec2 --dataset asvspoof [--batch-size 16] [--limit 200]
"""
import argparse
import hashlib
import json
import math
import os
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

DATASET_PATHS = {
    name: {
        "manifest": Path(f"data/manifest{suffix}.csv"),
        "mfcc_out": Path(f"data/features_mfcc{suffix}.csv"),
        "w2v_meta_out": Path(f"data/features_wav2vec2_meta{suffix}.csv"),
        "w2v_layers_out": Path(f"data/features_wav2vec2_layers{suffix}" + ("" if name == "asvspoof2021" else ".npy")),
    }
    for name, suffix in (("deepvoice", ""), ("asvspoof", "_asvspoof"), ("asvspoof2021", "_asvspoof2021"), ("pt", "_pt"))
}

# Aliases para o dataset padrão (DEEP-VOICE), usados por scripts que só fazem
# sentido nesse dataset: latency_bench.py, live_demo.py, speaker_dependent_eval.py,
# own_voice_ceiling.py, own_voice_pretrain_test.py.
MANIFEST_PATH = DATASET_PATHS["deepvoice"]["manifest"]
MFCC_OUT = DATASET_PATHS["deepvoice"]["mfcc_out"]

SR = 16000
N_MFCC = 20
# Colunas de metadado copiadas do manifest para os arquivos de atributos (as que
# existirem em cada dataset): tudo que não é atributo.
META_COLS = [
    "filepath", "speakers", "label", "channel", "source_file", "window_index",
    "split", "attack_id", "generator",
]

MFCC_COLUMNS = [
    "chroma_stft", "rms", "spectral_centroid", "spectral_bandwidth",
    "rolloff", "zero_crossing_rate",
] + [f"mfcc{i}" for i in range(1, N_MFCC + 1)]

W2V_MODEL_NAME = "facebook/wav2vec2-base"
W2V_N_LAYERS = 13  # hidden_states: 0 = entrada do transformer, 1..12 = saída de cada camada
W2V_DIM = 768
# Camada usada pelos classificadores Wav2Vec2 do projeto. Escolhida por
# src/layer_sweep.py olhando SÓ o split de validação (dev) do ASVspoof — nunca o
# teste nem o LOSO do DEEP-VOICE —, para a escolha não "espiar" o resultado final.
# Resultado (results/layer_sweep.csv, regime augmented): camada 4, EER médio de
# 0,093 no dev, contra 0,139 da camada 12 (a última, usada na primeira versão).
W2V_LAYER = 4
# Camada usada na primeira versão do projeto. Os testes externos também avaliam
# modelos com ela, para mostrar se a escolha feita no dev se sustenta fora dele.
W2V_ORIGINAL_LAYER = 12


def meta_cols_for(df: pd.DataFrame) -> list[str]:
    return [c for c in META_COLS if c in df.columns]


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


def run_mfcc(manifest: pd.DataFrame, mfcc_out: Path) -> None:
    with Pool(os.cpu_count() or 4) as pool:
        rows = list(tqdm(pool.imap(extract_mfcc, manifest["filepath"], chunksize=256), total=len(manifest), desc="MFCC"))
    feat = pd.DataFrame(rows, columns=MFCC_COLUMNS)
    out = pd.concat([manifest[meta_cols_for(manifest)].reset_index(drop=True), feat], axis=1)
    out.to_csv(mfcc_out, index=False)
    print(f"MFCC salvo em {mfcc_out}: {len(out)} janelas, {len(MFCC_COLUMNS)} atributos.")


def load_mfcc(dataset: str) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(DATASET_PATHS[dataset]["mfcc_out"])
    return df[meta_cols_for(df)], df[MFCC_COLUMNS].values


def load_wav2vec2_model(n_layers: int | None = None):
    """Carrega o Wav2Vec2 pré-treinado. Com n_layers, descarta as camadas do
    transformer acima dela — a rede faz menos contas e a saída da última camada
    mantida é exatamente a mesma da rede completa (usado no teste de latência)."""
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    processor = Wav2Vec2FeatureExtractor.from_pretrained(W2V_MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(W2V_MODEL_NAME).eval()
    if n_layers is not None:
        model.encoder.layers = model.encoder.layers[:n_layers]
    return processor, model


def wav2vec2_layer_embeddings(audios: list[np.ndarray], processor, model) -> np.ndarray:
    """Média no tempo de cada camada, para um lote de janelas: (camadas, lote, 768)."""
    import torch

    inputs = processor(audios, sampling_rate=SR, return_tensors="pt", padding=True)
    with torch.inference_mode():
        out = model(inputs.input_values, output_hidden_states=True)
    return torch.stack(out.hidden_states).mean(dim=2).numpy()


def stored_layers_path(layers_out: Path) -> Path:
    """Arquivo que diz quais camadas foram guardadas (ao lado do .npy, ou dentro da pasta de blocos)."""
    return layers_out / "layers.json" if layers_out.suffix != ".npy" else layers_out.with_suffix(".layers.json")


def run_wav2vec2(
    manifest: pd.DataFrame, layers_out: Path, meta_out: Path,
    batch_size: int, save_every: int, limit: int | None, threads: int, layers: list[int] | None = None,
) -> None:
    """Extrai as camadas pedidas (padrão: as 13) para um arquivo .npy
    (camadas, janelas, 768) em float16, escrito direto no disco (memmap) — cabe
    em pouca RAM mesmo com centenas de milhares de janelas. O progresso fica num
    .progress.json ao lado; uma execução interrompida retoma de onde parou,
    desde que o manifest seja o mesmo (mesmos arquivos, na mesma ordem)."""
    import torch

    torch.set_num_threads(threads)
    layers = list(range(W2V_N_LAYERS)) if layers is None else sorted(layers)
    rows = manifest if limit is None else manifest.head(limit)
    paths = rows["filepath"].tolist()
    n = len(paths)
    progress_path = layers_out.with_suffix(".progress.json")
    signature = hashlib.md5(("\n".join(paths) + f"|camadas={layers}").encode()).hexdigest()

    done = 0
    if layers_out.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress["signature"] == signature:
            done = progress["n_done"]
        else:
            print("Aviso: atributos existentes não batem com o manifest atual — recalculando do zero.")
    if done > 0:
        arr = np.load(layers_out, mmap_mode="r+")
    else:
        arr = np.lib.format.open_memmap(layers_out, mode="w+", dtype=np.float16, shape=(len(layers), n, W2V_DIM))
        rows[meta_cols_for(rows)].to_csv(meta_out, index=False)
        stored_layers_path(layers_out).write_text(json.dumps(layers))

    processor, model = load_wav2vec2_model()
    since_last_save = 0
    for i in tqdm(range(done, n, batch_size), desc="Wav2Vec2", initial=done // batch_size, total=math.ceil(n / batch_size)):
        batch_paths = paths[i:i + batch_size]
        audios = [librosa.load(p, sr=SR, mono=True)[0] for p in batch_paths]
        arr[:, i:i + len(batch_paths), :] = wav2vec2_layer_embeddings(audios, processor, model)[layers]
        since_last_save += len(batch_paths)

        if since_last_save >= save_every or i + batch_size >= n:
            arr.flush()
            progress_path.write_text(json.dumps({"signature": signature, "n_done": i + len(batch_paths), "n_total": n}))
            since_last_save = 0

    print(f"Wav2Vec2 salvo em {layers_out}: {n} janelas x camadas {layers} x {W2V_DIM} dimensões.")


def load_wav2vec2(dataset: str, layer: int = W2V_LAYER) -> tuple[pd.DataFrame, np.ndarray]:
    """Metadados + embeddings de UMA camada (float32), lidos do .npy sem carregar
    as outras camadas na memória. O ASVspoof 2021 é salvo em blocos (uma pasta
    de part_*.npy, ver build_dataset_asvspoof2021.py), concatenados aqui."""
    paths = DATASET_PATHS[dataset]
    meta = pd.read_csv(paths["w2v_meta_out"])
    layers_path = paths["w2v_layers_out"]
    layers_file = stored_layers_path(layers_path)
    stored = json.loads(layers_file.read_text()) if layers_file.exists() else list(range(W2V_N_LAYERS))
    if layer not in stored:
        raise ValueError(f"A camada {layer} não foi extraída para {dataset} (camadas guardadas: {stored}).")
    parts = sorted(layers_path.glob("part_*[0-9].npy")) if layers_path.is_dir() else [layers_path]
    X = np.concatenate([np.asarray(np.load(p, mmap_mode="r")[stored.index(layer)], dtype=np.float32) for p in parts])
    return meta, X


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["mfcc", "wav2vec2"], required=True)
    parser.add_argument("--dataset", choices=list(DATASET_PATHS), default="deepvoice")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=800)
    parser.add_argument("--threads", type=int, default=4, help="threads do PyTorch (4 = núcleos físicos deste notebook)")
    parser.add_argument("--limit", type=int, default=None, help="processa só as N primeiras linhas (teste/benchmark)")
    parser.add_argument("--layers", type=int, nargs="+", default=None, help="camadas a guardar (padrão: as 13)")
    args = parser.parse_args()

    paths = DATASET_PATHS[args.dataset]
    manifest = pd.read_csv(paths["manifest"])
    if args.kind == "mfcc":
        run_mfcc(manifest, paths["mfcc_out"])
    else:
        run_wav2vec2(
            manifest, paths["w2v_layers_out"], paths["w2v_meta_out"],
            batch_size=args.batch_size, save_every=args.save_every, limit=args.limit, threads=args.threads,
            layers=args.layers,
        )


if __name__ == "__main__":
    main()
