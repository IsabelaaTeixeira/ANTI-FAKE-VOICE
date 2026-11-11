"""Extrai os atributos do ASVspoof 2021 LA (subconjunto oficial "eval", 148 mil
gravações) — o teste em telefonia REAL do projeto.

No ASVspoof 2021 LA, os mesmos ataques e locutores do eval de 2019 foram
TRANSMITIDOS de verdade por sistemas de telefonia (7 condições: sem codec,
a-law e µ-law (VoIP/rede fixa), G.722 (VoIP banda larga), GSM, Opus e PSTN),
em ligações reais entre diferentes países. Aqui o áudio NÃO passa pelos
simuladores do projeto — ele já vem degradado pelo canal real. A pergunta é
se um modelo treinado com os canais SIMULADOS vai melhor nesses canais reais
do que um treinado só com áudio limpo (ver src/external_eval.py).

O pacote do eval tem 7,8 GB. Para não precisar do dobro disso em disco, o
.tar.gz é lido em sequência (streaming), e cada gravação passa pelo mesmo
preparo do resto do projeto em memória — remove silêncio de borda, fatia em
janelas de 1s, normaliza cada janela a -16 LUFS com o mesmo filtro do FFmpeg —
e vira atributo (MFCC + Wav2Vec2) sem que nenhuma janela de áudio seja gravada
em disco. Do Wav2Vec2 guarda só duas camadas (LAYERS): a escolhida para o
projeto (W2V_LAYER, ver src/layer_sweep.py — rode-o antes) e a 12, a usada
originalmente; as 13 camadas das ~233 mil janelas não caberiam no disco. Os atributos são salvos em blocos de
CHUNK_UTTERANCES gravações; uma execução interrompida retoma do último bloco
completo.

Gravações com menos de 1s de fala depois do corte de silêncio não geram
janela e não aparecem nos atributos: são 10.680 das 148.176 (7,2%, na mesma
proporção entre bonafide e spoof), então os CSVs têm 137.496 gravações. Para
comparar com os baselines oficiais nas mesmas gravações, ver
src/external_eval.baselines_2021.

Pré-requisitos (ver README):
    data/raw/ASVSPOOF2021/ASVspoof2021_LA_eval.tar.gz   (Zenodo, registro 4837263)
    data/raw/ASVSPOOF2021/keys/LA/CM/trial_metadata.txt (LA-keys-full.tar.gz, asvspoof.org)

Uso:
    python -m src.build_dataset_asvspoof2021
"""
import json
import subprocess
import tarfile
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src import channel_sim
from src.build_dataset import SILENCE_TOP_DB, WINDOW_SEC
from src.features import (
    DATASET_PATHS, MFCC_COLUMNS, SR, W2V_LAYER, extract_mfcc_from_array, load_wav2vec2_model,
    stored_layers_path, wav2vec2_layer_embeddings,
)

RAW_DIR = Path("data/raw/ASVSPOOF2021")
TAR_PATH = RAW_DIR / "ASVspoof2021_LA_eval.tar.gz"
METADATA_PATH = RAW_DIR / "keys/LA/CM/trial_metadata.txt"
PARTS_DIR = DATASET_PATHS["asvspoof2021"]["w2v_layers_out"]
CHUNK_UTTERANCES = 2000
N_WORKERS = 8
TORCH_THREADS = 4
BATCH_SIZE = 16
LAYERS = sorted({W2V_LAYER, 12})


def load_metadata() -> pd.DataFrame:
    meta = pd.read_csv(
        METADATA_PATH, sep=" ", header=None,
        names=["speaker", "trial", "codec", "transmission", "attack_id", "key", "trim", "subset"],
    )
    return meta[meta["subset"] == "eval"].set_index("trial")


def normalize_window(window: np.ndarray) -> np.ndarray:
    """Mesmo filtro de channel_sim.normalize_clean (loudnorm a -16 LUFS), via
    pipes do FFmpeg em vez de arquivos — PCM 16 bits na entrada e na saída,
    como o WAV que normalize_clean lê e escreve."""
    pcm = (np.clip(window, -1, 1) * 32767).astype("<i2").tobytes()
    out = subprocess.run([
        str(channel_sim.FFMPEG), "-v", "error", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0",
        "-ar", str(SR), "-ac", "1", "-t", "1.0", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-f", "s16le", "pipe:1",
    ], input=pcm, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype="<i2").astype(np.float32) / 32768


def prepare_utterance(job: tuple[str, bytes]) -> tuple[str, list[np.ndarray]]:
    """flac (bytes) -> janelas de 1s normalizadas, como build_dataset.slice_to_windows."""
    trial, flac_bytes = job
    # decodifica com o FFmpeg: a libsndfile 1.2.2 falha ("flac decoder lost sync")
    # em parte dos FLACs do ASVspoof 2021, que o FFmpeg lê sem erro
    pcm = subprocess.run([
        str(channel_sim.FFMPEG), "-v", "error", "-i", "pipe:0", "-f", "s16le", "-ar", str(SR), "-ac", "1", "pipe:1",
    ], input=flac_bytes, capture_output=True, check=True).stdout
    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
    audio, _ = librosa.effects.trim(audio, top_db=SILENCE_TOP_DB)
    window_len = int(WINDOW_SEC * SR)
    windows = [normalize_window(audio[i * window_len:(i + 1) * window_len]) for i in range(len(audio) // window_len)]
    return trial, windows


def iter_eval_chunks(eval_trials: set[str], skip: int):
    """Percorre o .tar.gz em sequência e entrega listas de (trial, bytes) de
    CHUNK_UTTERANCES gravações do subconjunto eval, pulando as `skip` primeiras."""
    seen = 0
    chunk = []
    with tarfile.open(TAR_PATH, "r|gz") as tar:
        for member in tar:
            trial = Path(member.name).stem
            if not member.isfile() or not member.name.endswith(".flac") or trial not in eval_trials:
                continue
            seen += 1
            if seen <= skip:
                continue
            chunk.append((trial, tar.extractfile(member).read()))
            if len(chunk) == CHUNK_UTTERANCES:
                yield chunk
                chunk = []
    if chunk:
        yield chunk


def process_chunk(chunk, meta: pd.DataFrame, pool: Pool, processor, model, part_index: int) -> None:
    prepared = pool.map(prepare_utterance, chunk, chunksize=16)
    rows, windows = [], []
    for trial, utt_windows in prepared:
        info = meta.loc[trial]
        for idx, window in enumerate(utt_windows):
            rows.append({
                "speakers": info["speaker"], "label": "real" if info["key"] == "bonafide" else "fake",
                "channel": info["codec"], "transmission": info["transmission"], "attack_id": info["attack_id"],
                "source_file": trial, "window_index": idx,
            })
            windows.append(window)

    mfcc = pool.map(extract_mfcc_from_array, windows, chunksize=64)
    layers = np.concatenate([
        wav2vec2_layer_embeddings(windows[i:i + BATCH_SIZE], processor, model)[LAYERS]
        for i in range(0, len(windows), BATCH_SIZE)
    ], axis=1).astype(np.float16) if windows else np.zeros((len(LAYERS), 0, 768), dtype=np.float16)

    stem = f"part_{part_index:05d}"
    np.save(PARTS_DIR / f"{stem}.tmp.npy", layers)
    (PARTS_DIR / f"{stem}.tmp.npy").rename(PARTS_DIR / f"{stem}.npy")
    meta_df = pd.concat([pd.DataFrame(rows), pd.DataFrame(mfcc, columns=MFCC_COLUMNS)], axis=1)
    meta_df.to_csv(PARTS_DIR / f"{stem}.csv", index=False)  # escrito por último: marca o bloco como completo


def merge_parts() -> None:
    """Junta os metadados+MFCC de todos os blocos nos CSVs finais do dataset."""
    parts = sorted(PARTS_DIR.glob("part_*.csv"))
    full = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    full.to_csv(DATASET_PATHS["asvspoof2021"]["mfcc_out"], index=False)
    full.drop(columns=MFCC_COLUMNS).to_csv(DATASET_PATHS["asvspoof2021"]["w2v_meta_out"], index=False)
    print(f"{len(full)} janelas de {full['source_file'].nunique()} gravações.")
    print(full.groupby(["channel", "label"]).size())


def main() -> None:
    torch.set_num_threads(TORCH_THREADS)
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    layers_file = stored_layers_path(PARTS_DIR)
    if layers_file.exists() and json.loads(layers_file.read_text()) != LAYERS:
        raise SystemExit(f"Os blocos em {PARTS_DIR} têm outras camadas ({layers_file.read_text()}); apague a pasta para refazer com {LAYERS}.")
    layers_file.write_text(json.dumps(LAYERS))
    meta = load_metadata()
    parts_done = len(list(PARTS_DIR.glob("part_*.csv")))
    skip = parts_done * CHUNK_UTTERANCES
    n_chunks = int(np.ceil(len(meta) / CHUNK_UTTERANCES))
    print(f"{len(meta)} gravações no eval; {parts_done} blocos já prontos, retomando a partir da gravação {skip}.")

    # o pool é criado ANTES de carregar o PyTorch: processos filhos herdando o
    # estado de threads do PyTorch via fork podem travar
    with Pool(N_WORKERS) as pool:
        processor, model = load_wav2vec2_model()
        for i, chunk in enumerate(tqdm(iter_eval_chunks(set(meta.index), skip), total=n_chunks, initial=parts_done, desc="ASVspoof 2021 (blocos)")):
            process_chunk(chunk, meta, pool, processor, model, parts_done + i)
    merge_parts()


if __name__ == "__main__":
    main()
