"""Fatia o DEEP-VOICE em janelas de 1s, aplica os canais simulados e gera o
manifest (data/manifest.csv: filepath, speakers, label, channel, source_file,
window_index).

`speakers` é a coluna usada para o split Leave-One-Speaker-Out: para REAL é o
próprio locutor; para FAKE é "origem,destino" (as duas identidades presentes
no arquivo), já que um fold que isola um locutor precisa excluir tanto as
gravações reais dele quanto qualquer conversão onde ele aparece como origem
ou destino.

Canais de codec (gsm, volte, vonr, vonr_loss) são aplicados na gravação
INTEIRA e só depois fatiados — o codec roda contínuo, como numa ligação. O
canal mic é aplicado janela a janela: não é um codec com memória do sinal, e
sortear uma sala/microfone por janela dá mais variedade de ambientes.

`vonr_loss` (VoNR com perda de pacotes) é um canal só de TESTE: train.py
não o usa no treino, para medir a robustez a uma degradação nunca vista.

Uso:
    python -m src.build_dataset --channels clean gsm volte vonr mic vonr_loss
"""
import argparse
import os
import re
import tempfile
import zlib
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from src import channel_sim, mic_sim

RAW_DIR = Path("data/raw/KAGGLE/AUDIO")
WINDOWS_DIR = Path("data/raw/windows")
CHANNELED_DIR = Path("data/channeled")
MANIFEST_PATH = Path("data/manifest.csv")

SR = 16000
WINDOW_SEC = 1.0
ALL_CHANNELS = ["clean", "gsm", "volte", "vonr", "mic", "vonr_loss"]
N_WORKERS = os.cpu_count() or 4

# Locutores excluídos do estudo (decisão do projeto, não técnica).
EXCLUDED_SPEAKERS = {"musk", "trump"}


def _file_seed(path: Path) -> int:
    # seed determinístico por arquivo: reprodutível, mas varia entre arquivos/janelas
    return zlib.crc32(str(path).encode()) & 0xFFFFFFFF


# Canais de codec: recebem a gravação inteira (src) e escrevem a versão decodificada (dst).
CODEC_FN = {
    "gsm": lambda src, dst, seed: channel_sim.to_gsm(src, dst),
    "volte": lambda src, dst, seed: channel_sim.to_volte(src, dst),
    "vonr": lambda src, dst, seed: channel_sim.to_vonr(src, dst),
    "vonr_loss": lambda src, dst, seed: channel_sim.to_vonr_loss(src, dst, seed=seed),
}


def parse_speakers(filename: str, label: str) -> str:
    stem = Path(filename).stem
    if label == "REAL":
        return stem.replace("-original", "").lower()
    match = re.match(r"^(.+)-to-(.+)$", stem, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Nome de arquivo FAKE fora do padrão 'origem-to-destino': {filename}")
    source, target = match.group(1).lower(), match.group(2).lower()
    return ",".join(sorted({source, target}))


SILENCE_TOP_DB = 30  # mesmo limiar usado em audit_silence.py (fase T)


def load_trimmed(audio_path: Path) -> np.ndarray:
    """Carrega a gravação a 16kHz mono e remove o silêncio de borda (início/fim)."""
    audio, _ = librosa.load(audio_path, sr=SR, mono=True)
    audio, _ = librosa.effects.trim(audio, top_db=SILENCE_TOP_DB)
    return audio


def apply_codec_to_recording(audio: np.ndarray, channel: str, seed: int, tmp_dir: Path) -> np.ndarray:
    """Passa a gravação inteira pelo codec do canal e devolve o áudio decodificado,
    já sem o atraso algorítmico do codec e com o mesmo comprimento da entrada."""
    full_in = tmp_dir / "full_in.wav"
    full_out = tmp_dir / "full_out.wav"
    sf.write(full_in, audio, SR)
    CODEC_FN[channel](full_in, full_out, seed)
    coded, _ = sf.read(full_out, dtype="float32")
    coded = coded[channel_sim.CODEC_DELAY_SAMPLES[channel]:]
    return np.pad(coded, (0, max(0, len(audio) - len(coded))))[: len(audio)]


def slice_to_windows(
    audio_path: Path, out_dir: Path, channel: str = "clean", expected: int | None = None,
) -> list[Path]:
    """Corta a gravação em janelas de 1s. Antes de fatiar, remove o silêncio de
    borda da gravação inteira e, nos canais de codec, passa a gravação inteira
    pelo codec. Cada janela sai normalizada a -16 LUFS, em qualquer canal —
    corrige o atalho de silêncio/volume encontrado na auditoria da fase T.

    Se as janelas já existem, são reaproveitadas — a menos que `expected` (o
    número de janelas do canal limpo) não bata, sinal de uma execução
    interrompida no meio dessa gravação: aí ela é refeita do zero."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem
    existing = sorted(out_dir.glob(f"{stem}_*.wav"))
    if existing and (expected is None or len(existing) == expected):
        return existing
    for stale in existing:
        stale.unlink()

    audio = load_trimmed(audio_path)
    window_len = int(WINDOW_SEC * SR)
    n_windows = len(audio) // window_len

    paths = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        if channel != "clean" and n_windows > 0:
            audio = apply_codec_to_recording(audio, channel, _file_seed(audio_path), tmp_dir)
        for i in range(n_windows):
            window = audio[i * window_len : (i + 1) * window_len]
            raw_path = tmp_dir / f"{stem}_{i:05d}.wav"
            out_path = out_dir / f"{stem}_{i:05d}.wav"
            sf.write(raw_path, window, SR)
            channel_sim.normalize_clean(raw_path, out_path)
            paths.append(out_path)
    return paths


def build_clean_windows() -> tuple[list[dict], dict[str, Path]]:
    rows, source_paths = [], {}
    for label, subdir in (("REAL", "REAL"), ("FAKE", "FAKE")):
        src_dir = RAW_DIR / subdir
        for audio_path in tqdm(sorted(src_dir.glob("*.wav")), desc=f"Fatiando {label}"):
            speakers = parse_speakers(audio_path.name, label)
            if set(speakers.split(",")) & EXCLUDED_SPEAKERS:
                continue
            source_paths[audio_path.name] = audio_path
            windows = slice_to_windows(audio_path, WINDOWS_DIR / label)
            for idx, window_path in enumerate(windows):
                rows.append({
                    "filepath": str(window_path),
                    "speakers": speakers,
                    "label": label.lower(),
                    "channel": "clean",
                    "source_file": audio_path.name,
                    "window_index": idx,
                })
    return rows, source_paths


def _codec_job(job: tuple[Path, Path, int], channel: str) -> list[Path]:
    audio_path, out_dir, expected = job
    return slice_to_windows(audio_path, out_dir, channel, expected)


def _mic_job(job: tuple[Path, Path]) -> None:
    clean_path, out_path = job
    if not out_path.exists():
        mic_sim.to_mic(clean_path, out_path, seed=_file_seed(clean_path))


def apply_channel(
    channel: str, clean_rows: list[dict], source_paths: dict[str, Path], channeled_dir: Path,
) -> list[dict]:
    """Gera as janelas de um canal para todas as janelas limpas em clean_rows,
    em paralelo. A janela i do canal corresponde à janela i do canal limpo da
    mesma gravação (mesmo trecho de fala)."""
    def out_path_for(row: dict) -> Path:
        return channeled_dir / channel / row["label"] / Path(row["filepath"]).name

    if channel == "mic":
        jobs = [(Path(r["filepath"]), out_path_for(r)) for r in clean_rows]
        for _, out_path in jobs:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        with Pool(N_WORKERS) as pool:
            list(tqdm(pool.imap_unordered(_mic_job, jobs, chunksize=64), total=len(jobs), desc="Aplicando canal mic"))
    else:
        n_clean = pd.Series([r["source_file"] for r in clean_rows]).value_counts()
        jobs = {}
        for r in clean_rows:
            jobs[r["source_file"]] = (
                source_paths[r["source_file"]], channeled_dir / channel / r["label"], int(n_clean[r["source_file"]]),
            )
        with Pool(N_WORKERS) as pool:
            list(tqdm(
                pool.imap_unordered(partial(_codec_job, channel=channel), jobs.values()),
                total=len(jobs), desc=f"Aplicando canal {channel}",
            ))

    out_rows = []
    for row in clean_rows:
        out_path = out_path_for(row)
        if not out_path.exists():
            raise RuntimeError(f"Janela do canal {channel} não foi gerada: {out_path}")
        out_rows.append({**row, "filepath": str(out_path), "channel": channel})
    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", nargs="+", default=["clean"], choices=ALL_CHANNELS)
    args = parser.parse_args()

    clean_rows, source_paths = build_clean_windows()
    all_rows = list(clean_rows) if "clean" in args.channels else []

    for channel in args.channels:
        if channel == "clean":
            continue
        all_rows.extend(apply_channel(channel, clean_rows, source_paths, CHANNELED_DIR))

    df = pd.DataFrame(all_rows)
    df.to_csv(MANIFEST_PATH, index=False)
    print(f"\nManifest salvo em {MANIFEST_PATH}: {len(df)} janelas.")
    print(df.groupby(["channel", "label"]).size())


if __name__ == "__main__":
    main()
