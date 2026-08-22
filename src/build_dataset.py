"""Fatia o DEEP-VOICE em janelas de 1s, aplica os canais simulados e gera o
manifest (data/manifest.csv: filepath, speakers, label, channel, source_file,
window_index).

`speakers` é a coluna usada para o split Leave-One-Speaker-Out: para REAL é o
próprio locutor; para FAKE é "origem,destino" (as duas identidades presentes
no arquivo), já que um fold que isola um locutor precisa excluir tanto as
gravações reais dele quanto qualquer conversão onde ele aparece como origem
ou destino.

Uso:
    python -m src.build_dataset --channels clean
    python -m src.build_dataset --channels clean gsm volte   # após corrigir o ffmpeg
    python -m src.build_dataset --channels clean gsm volte vonr mic
"""
import argparse
import re
import tempfile
import zlib
from pathlib import Path

import soundfile as sf
import librosa
import pandas as pd
from tqdm import tqdm

from src import channel_sim, mic_sim

RAW_DIR = Path("data/raw/KAGGLE/AUDIO")
WINDOWS_DIR = Path("data/raw/windows")
CHANNELED_DIR = Path("data/channeled")
MANIFEST_PATH = Path("data/manifest.csv")

SR = 16000
WINDOW_SEC = 1.0

# Locutores excluídos do estudo (decisão do projeto, não técnica).
EXCLUDED_SPEAKERS = {"musk", "trump"}


def _mic_channel(src: Path, dst: Path) -> None:
    # seed determinístico por arquivo (reprodutível, mas varia a sala/mic simulados entre janelas)
    seed = zlib.crc32(str(src).encode()) & 0xFFFFFFFF
    mic_sim.to_mic(src, dst, seed=seed)


CHANNEL_FN = {
    "gsm": channel_sim.to_gsm,
    "volte": channel_sim.to_volte,
    "vonr": channel_sim.to_vonr,
    "mic": _mic_channel,
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


def slice_to_windows(audio_path: Path, out_dir: Path) -> list[Path]:
    """Corta o áudio em janelas de 1s. Antes de fatiar, remove o silêncio de
    borda (início/fim) da gravação inteira, e cada janela sai normalizada a
    -16 LUFS (mesma normalização já aplicada em gsm/volte/vonr) — corrige o
    atalho de silêncio/volume encontrado na auditoria da fase T."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem
    existing = sorted(out_dir.glob(f"{stem}_*.wav"))
    if existing:
        return existing

    audio, _ = librosa.load(audio_path, sr=SR, mono=True)
    audio, _ = librosa.effects.trim(audio, top_db=SILENCE_TOP_DB)
    window_len = int(WINDOW_SEC * SR)
    n_windows = len(audio) // window_len

    paths = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i in range(n_windows):
            window = audio[i * window_len : (i + 1) * window_len]
            raw_path = Path(tmp_dir) / f"{stem}_{i:05d}.wav"
            out_path = out_dir / f"{stem}_{i:05d}.wav"
            sf.write(raw_path, window, SR)
            channel_sim.normalize_clean(raw_path, out_path)
            paths.append(out_path)
    return paths


def build_clean_windows() -> list[dict]:
    rows = []
    for label, subdir in (("REAL", "REAL"), ("FAKE", "FAKE")):
        src_dir = RAW_DIR / subdir
        for audio_path in tqdm(sorted(src_dir.glob("*.wav")), desc=f"Fatiando {label}"):
            speakers = parse_speakers(audio_path.name, label)
            if set(speakers.split(",")) & EXCLUDED_SPEAKERS:
                continue
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
    return rows


def apply_channel(channel: str, clean_rows: list[dict]) -> list[dict]:
    fn = CHANNEL_FN[channel]
    out_rows = []
    for row in tqdm(clean_rows, desc=f"Aplicando canal {channel}"):
        clean_path = Path(row["filepath"])
        out_dir = CHANNELED_DIR / channel / row["label"]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / clean_path.name
        if not out_path.exists():
            fn(clean_path, out_path)
        out_rows.append({**row, "filepath": str(out_path), "channel": channel})
    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--channels", nargs="+", default=["clean"],
        choices=["clean", "gsm", "volte", "vonr", "mic"],
    )
    args = parser.parse_args()

    clean_rows = build_clean_windows()
    all_rows = list(clean_rows) if "clean" in args.channels else []

    for channel in args.channels:
        if channel == "clean":
            continue
        all_rows.extend(apply_channel(channel, clean_rows))

    df = pd.DataFrame(all_rows)
    df.to_csv(MANIFEST_PATH, index=False)
    print(f"\nManifest salvo em {MANIFEST_PATH}: {len(df)} janelas.")
    print(df.groupby(["channel", "label"]).size())


if __name__ == "__main__":
    main()
