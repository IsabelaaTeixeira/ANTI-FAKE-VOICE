"""Atributos do split de treino COMPLETO do ASVspoof 2019 LA (25.380 gravações,
20 locutores, ataques A01-A06), nos 5 canais de treino.

O resto do projeto usa uma amostra do treino (até 20 gravações por
combinação locutor x ataque: 2.683 gravações, ver build_dataset_asvspoof.py).
Aqui entra o split inteiro, o mesmo usado pelos detectores de referência do
desafio. Fatiar e gravar em disco as janelas de 25 mil gravações em 5 canais
ocuparia ~7 GB; por isso cada gravação é preparada em memória, exatamente como
em build_dataset.slice_to_windows (corte do silêncio de borda, codec na
gravação inteira com o atraso compensado, janelas de 1 s normalizadas a
-16 LUFS; o canal mic é aplicado à janela limpa já normalizada, com a mesma
semente por janela), e só os atributos do Wav2Vec2 (camada W2V_LAYER) vão
para o disco: a média no tempo (768 valores) e a sequência projetada pela PCA
(49 x 128), ver src/sequence_features.py.

Etapas:
    python -m src.build_dataset_asvspoof_full fit-pca  # PCA com janelas da amostra de treino (já em disco)
    python -m src.build_dataset_asvspoof_full build    # treino completo, em blocos (retoma se interrompido)
    python -m src.build_dataset_asvspoof_full dev      # dev da amostra, canais de treino (parada antecipada)
    python -m src.build_dataset_asvspoof_full check    # confere contra os atributos já extraídos da amostra
"""
import argparse
import tempfile
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from src import channel_sim, mic_sim
from src.build_dataset import _file_seed, apply_codec_to_recording, load_trimmed
from src.build_dataset_asvspoof import RAW_DIR, SPLIT_DIRS, WINDOWS_DIR, load_protocol
from src.features import SR
from src.train import TRAIN_CHANNELS

FULL_DIR = Path("data/features_seq_asvspoof_full")
DEV_DIR = Path("data/features_seq_asvspoof_dev")
MANIFEST_ASVSPOOF = Path("data/manifest_asvspoof.csv")
BLOCK_RECORDINGS = 300
N_WORKERS = 6
TORCH_THREADS = 4
PCA_WINDOWS_PER_CHANNEL = 800
WINDOW = SR  # 1 s
CODEC_CHANNELS = ("gsm", "volte", "vonr")


def normalize(window: np.ndarray, tmp_dir: Path) -> np.ndarray:
    """Mesma normalização de build_dataset.slice_to_windows, pelo mesmo caminho
    (WAV de 16 bits -> channel_sim.normalize_clean -> WAV), para as janelas
    saírem idênticas às do pipeline em disco."""
    raw, out = tmp_dir / "win_raw.wav", tmp_dir / "win_norm.wav"
    sf.write(raw, window, SR)
    channel_sim.normalize_clean(raw, out)
    return sf.read(out, dtype="float32")[0]


def prepare_recording(job: tuple[str, dict]) -> tuple[dict, dict[str, list[np.ndarray]]]:
    """Gravação -> janelas de 1 s normalizadas nos 5 canais de treino, em memória."""
    flac_path, info = job
    audio_path = Path(flac_path)
    audio = load_trimmed(audio_path)
    n = len(audio) // WINDOW
    if n == 0:
        return info, {}
    windows = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        windows["clean"] = [normalize(audio[i * WINDOW:(i + 1) * WINDOW], tmp_dir) for i in range(n)]
        for channel in CODEC_CHANNELS:
            coded = apply_codec_to_recording(audio, channel, _file_seed(audio_path), tmp_dir)
            windows[channel] = [normalize(coded[i * WINDOW:(i + 1) * WINDOW], tmp_dir) for i in range(n)]
        mic = []
        for i, clean in enumerate(windows["clean"]):
            # mesma semente que build_dataset.apply_channel usaria: a do caminho da janela limpa
            canonical = WINDOWS_DIR / "train" / info["label"] / f"{audio_path.stem}_{i:05d}.wav"
            src, dst = tmp_dir / "mic_in.wav", tmp_dir / "mic_out.wav"
            sf.write(src, clean, SR)
            mic_sim.to_mic(src, dst, seed=_file_seed(canonical))
            mic.append(sf.read(dst, dtype="float32")[0][:WINDOW])
        windows["mic"] = mic
    return info, windows


def train_jobs() -> list[tuple[str, dict]]:
    protocol = load_protocol("train")
    flac_dir = RAW_DIR / SPLIT_DIRS["train"] / "flac"
    return [
        (str(flac_dir / f"{r.filename}.flac"),
         {"speakers": r.speaker, "label": r.label, "attack_id": r.attack_id, "source_file": f"{r.filename}.flac"})
        for r in protocol.itertuples()
    ]


def save_part(out_dir: Path, index: int, rows: list[dict], pooled: np.ndarray, seq: np.ndarray) -> None:
    stem = out_dir / f"part_{index:05d}"
    for name, arr in (("pooled", pooled), ("seq", seq)):
        tmp = Path(f"{stem}.{name}.tmp.npy")
        np.save(tmp, arr)
        tmp.rename(f"{stem}.{name}.npy")
    pd.DataFrame(rows).to_csv(f"{stem}.csv", index=False)  # escrito por último: marca o bloco como completo


def flatten(prepared: list[tuple[dict, dict]]) -> tuple[list[dict], list[np.ndarray]]:
    rows, audios = [], []
    for info, windows in prepared:
        for channel in TRAIN_CHANNELS:
            for idx, w in enumerate(windows.get(channel, [])):
                rows.append({**info, "channel": channel, "window_index": idx, "split": "train"})
                audios.append(w)
    return rows, audios


def build() -> None:
    import torch

    from src.sequence_features import encode_windows, load_layer_model, load_pca

    torch.set_num_threads(TORCH_THREADS)
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    jobs = train_jobs()
    blocks = [jobs[i:i + BLOCK_RECORDINGS] for i in range(0, len(jobs), BLOCK_RECORDINGS)]
    done = len(list(FULL_DIR.glob("part_*.csv")))
    print(f"{len(jobs)} gravações de treino em {len(blocks)} blocos; {done} blocos já prontos.")
    if done >= len(blocks):
        return
    # o pool é criado antes de carregar a rede (processos filhos herdando threads do PyTorch podem travar)
    with Pool(N_WORKERS) as pool:
        processor, model = load_layer_model()
        pca = load_pca()
        pending = pool.map_async(prepare_recording, blocks[done], chunksize=4)
        for b in tqdm(range(done, len(blocks)), initial=done, total=len(blocks), desc="Treino completo (blocos)"):
            prepared = pending.get()
            if b + 1 < len(blocks):  # prepara o próximo bloco enquanto a rede processa este
                pending = pool.map_async(prepare_recording, blocks[b + 1], chunksize=4)
            rows, audios = flatten(prepared)
            pooled, seq = encode_windows(audios, processor, model, pca)
            save_part(FULL_DIR, b, rows, pooled, seq)


def encode_manifest_rows(rows: pd.DataFrame, out_dir: Path) -> None:
    """Atributos de janelas que já estão em disco (amostra do ASVspoof), num bloco só."""
    import torch

    from src.sequence_features import encode_windows, load_layer_model, load_pca

    torch.set_num_threads(TORCH_THREADS)
    out_dir.mkdir(parents=True, exist_ok=True)
    processor, model = load_layer_model()
    pca = load_pca()
    pooled, seq = [], []
    paths = rows["filepath"].tolist()
    for i in tqdm(range(0, len(paths), 512), desc=f"Atributos {out_dir.name}"):
        audios = [librosa.load(p, sr=SR)[0] for p in paths[i:i + 512]]
        p, s = encode_windows(audios, processor, model, pca)
        pooled.append(p)
        seq.append(s)
    meta = rows.drop(columns=["filepath"]).to_dict("records")
    save_part(out_dir, 0, meta, np.concatenate(pooled), np.concatenate(seq))


def fit_pca_from_sample() -> None:
    import torch

    from src.sequence_features import PCA_DIM, fit_pca, layer_frames, load_layer_model

    torch.set_num_threads(TORCH_THREADS)
    manifest = pd.read_csv(MANIFEST_ASVSPOOF)
    train = manifest[(manifest["split"] == "train") & manifest["channel"].isin(TRAIN_CHANNELS)]
    sample = train.groupby("channel", group_keys=False).sample(n=PCA_WINDOWS_PER_CHANNEL, random_state=42)
    processor, model = load_layer_model()
    frames = []
    paths = sample["filepath"].tolist()
    for i in tqdm(range(0, len(paths), 16), desc="Quadros para a PCA"):
        audios = [librosa.load(p, sr=SR)[0] for p in paths[i:i + 16]]
        frames.append(layer_frames(audios, processor, model).reshape(-1, 768))
    frames = np.concatenate(frames)
    params = fit_pca(frames)
    print(f"PCA ajustada em {len(frames)} quadros de {len(sample)} janelas de treino: "
          f"{PCA_DIM} componentes explicam {params['explained_variance_ratio'].sum():.1%} da variância.")


def check_against_sample() -> None:
    """As janelas da amostra de treino entram também no treino completo: a média
    calculada aqui tem de bater com o atributo já extraído pelo pipeline em disco."""
    from src.features import load_wav2vec2

    meta, X = load_wav2vec2("asvspoof")
    meta = meta.assign(row=np.arange(len(meta)))
    sample = meta[meta["split"] == "train"]
    parts = sorted(FULL_DIR.glob("part_*.csv"))
    full = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    pooled = np.concatenate([np.load(str(p).replace(".csv", ".pooled.npy")) for p in parts]).astype(np.float32)
    full = full.assign(row_full=np.arange(len(full)))
    m = sample.merge(full, on=["source_file", "channel", "window_index"])
    diff = np.abs(pooled[m["row_full"].values] - X[m["row"].values])
    scale = np.abs(X[m["row"].values]).mean()
    print(f"{len(m)} de {len(sample)} janelas da amostra encontradas no treino completo.")
    for channel, idx in m.groupby("channel").indices.items():
        print(f"  {channel:6s} dif. média {diff[idx].mean():.5f}  máx. {diff[idx].max():.4f}  (escala média {scale:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["fit-pca", "build", "dev", "check"])
    args = parser.parse_args()
    if args.step == "fit-pca":
        fit_pca_from_sample()
    elif args.step == "build":
        build()
    elif args.step == "dev":
        manifest = pd.read_csv(MANIFEST_ASVSPOOF)
        dev = manifest[(manifest["split"] == "dev") & manifest["channel"].isin(TRAIN_CHANNELS)].reset_index(drop=True)
        encode_manifest_rows(dev, DEV_DIR)
    else:
        check_against_sample()


if __name__ == "__main__":
    main()
