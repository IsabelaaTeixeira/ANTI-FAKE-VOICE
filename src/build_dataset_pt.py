"""Monta o conjunto de teste em português: vozes reais do Multilingual
LibriSpeech (MLS, splits dev+test, 20 locutores, CC BY 4.0) e clones delas
feitos por dois sintetizadores atuais (XTTS v2 e YourTTS), em todos os
canais simulados do projeto. Nenhum modelo do projeto é treinado com ele: é
um teste de generalização para outro idioma e para ataques mais recentes
que os do ASVspoof 2019 e do DEEP-VOICE (ver src/external_eval.py).

Desenho pareado:
  - cada locutor é clonado a partir de UMA gravação de referência dele
    (a de duração mais próxima de REFERENCE_TARGET_S), que fica fora do
    conjunto real de teste;
  - os dois sintetizadores falam as MESMAS frases (N_FAKE_PER_SPEAKER por
    locutor), tiradas de outras gravações do próprio locutor — real e clone
    dizem o mesmo tipo de conteúdo, e o detector não pode se apoiar nisso;
  - só entram frases de até MAX_TEXT_CHARS caracteres (o XTTS trunca o áudio
    de textos mais longos em português).

Compressão igualada: o MLS distribuído no Hugging Face guarda o áudio em
Opus (~35 kb/s). Se só a voz real tivesse artefatos de Opus, o detector
poderia aprender o atalho "tem Opus = real". Por isso cada voz clonada
também passa por Opus (libopus, OPUS_BITRATE, mesmo FFmpeg do sistema)
antes de ser fatiada — as duas classes chegam com o mesmo histórico de
compressão.

Ressalva: o MLS vem de audiolivros do LibriVox (gravações caseiras,
publicadas originalmente em MP3), com sotaques variados de português — a
voz real ainda carrega esse MP3 de origem, que a clonada não tem.

Três etapas, porque a clonagem roda no ambiente virtual separado do TTS:
    python -m src.build_dataset_pt prepare        # áudio real + lista de frases a clonar
    .venv-tts/bin/python src/clone_voice_pt.py    # gera as vozes clonadas
    python -m src.build_dataset_pt build          # janelas em todos os canais + manifest
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from src.build_dataset import ALL_CHANNELS, SR, apply_channel, slice_to_windows

RAW_DIR = Path("data/raw/PT")
PARQUETS = [RAW_DIR / "mls_pt_dev.parquet", RAW_DIR / "mls_pt_test.parquet"]
REAL_DIR = RAW_DIR / "real"
FAKE_DIR = RAW_DIR / "fake"
FAKE_OPUS_DIR = RAW_DIR / "fake_opus"
JOBS_PATH = RAW_DIR / "clone_jobs.csv"
WINDOWS_DIR = Path("data/raw/windows_pt")
CHANNELED_DIR = Path("data/channeled_pt")
MANIFEST_PATH = Path("data/manifest_pt.csv")

GENERATORS = ("xtts_v2", "your_tts")
N_FAKE_PER_SPEAKER = 25
MAX_TEXT_CHARS = 200
REFERENCE_TARGET_S = 12.0
RANDOM_SEED = 42
# FFmpeg do sistema (apt): tem o encoder libopus, que o build próprio com AMR não tem
SYSTEM_FFMPEG = "ffmpeg"
OPUS_BITRATE = "32k"


def decode_to_16k(data: bytes) -> np.ndarray:
    """Decodifica qualquer áudio (aqui, Opus a 48kHz) para 16kHz mono via FFmpeg."""
    pcm = subprocess.run(
        [SYSTEM_FFMPEG, "-v", "error", "-i", "pipe:0", "-f", "s16le", "-ar", str(SR), "-ac", "1", "pipe:1"],
        input=data, capture_output=True, check=True,
    ).stdout
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768


def opus_roundtrip(src: Path, dst: Path) -> None:
    """Codifica em Opus (mesma faixa de taxa do MLS) e decodifica de volta a 16kHz."""
    opus = subprocess.run(
        [SYSTEM_FFMPEG, "-v", "error", "-i", str(src), "-ac", "1", "-c:a", "libopus", "-b:a", OPUS_BITRATE, "-f", "ogg", "pipe:1"],
        capture_output=True, check=True,
    ).stdout
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, decode_to_16k(opus), SR)


def prepare() -> None:
    df = pd.concat([pd.read_parquet(p) for p in PARQUETS], ignore_index=True)
    rng = np.random.default_rng(RANDOM_SEED)
    jobs = []
    for speaker, g in tqdm(df.groupby("speaker_id"), desc="Locutores"):
        speaker = f"mls_{speaker}"
        out_dir = REAL_DIR / speaker
        out_dir.mkdir(parents=True, exist_ok=True)
        ref_idx = (g["audio_duration"] - REFERENCE_TARGET_S).abs().idxmin()
        for idx, row in g.iterrows():
            name = f"{row['id']}{'_referencia' if idx == ref_idx else ''}.wav"
            if not (out_dir / name).exists():
                sf.write(out_dir / name, decode_to_16k(row["audio"]["bytes"]), SR)
        reference = out_dir / f"{g.loc[ref_idx, 'id']}_referencia.wav"

        candidates = g.drop(index=ref_idx)
        candidates = candidates[candidates["transcript"].str.len() <= MAX_TEXT_CHARS]
        chosen = candidates.iloc[rng.permutation(len(candidates))[:N_FAKE_PER_SPEAKER]]
        for _, row in chosen.iterrows():
            jobs.append({
                "job_id": f"{speaker}_{row['id']}", "speaker": speaker,
                "reference_path": str(reference), "text": row["transcript"],
            })
    pd.DataFrame(jobs).to_csv(JOBS_PATH, index=False)
    print(f"{df['speaker_id'].nunique()} locutores, {len(df)} gravações reais; {len(jobs)} frases a clonar em {JOBS_PATH}")


def build(channels: list[str]) -> None:
    sources = []
    for path in sorted(REAL_DIR.glob("*/*.wav")):
        if "_referencia" not in path.stem:
            sources.append({"path": path, "speakers": path.parent.name, "label": "real", "generator": "bonafide"})
    jobs = pd.read_csv(JOBS_PATH).set_index("job_id")
    for generator in GENERATORS:
        for synth_path in sorted((FAKE_DIR / generator).glob("*.wav")):
            if synth_path.name.endswith(".tmp.wav"):  # síntese interrompida no meio
                continue
            path = FAKE_OPUS_DIR / generator / synth_path.name
            if not path.exists():
                opus_roundtrip(synth_path, path)
            sources.append({
                "path": path, "speakers": jobs.loc[path.stem.removeprefix(f"{generator}_"), "speaker"],
                "label": "fake", "generator": generator,
            })

    clean_rows, source_paths = [], {}
    for src in tqdm(sources, desc="Fatiando"):
        source_paths[src["path"].name] = src["path"]
        windows = slice_to_windows(src["path"], WINDOWS_DIR / src["label"])
        for idx, window_path in enumerate(windows):
            clean_rows.append({
                "filepath": str(window_path), "speakers": src["speakers"], "label": src["label"],
                "channel": "clean", "source_file": src["path"].name, "window_index": idx,
                "generator": src["generator"],
            })

    all_rows = list(clean_rows)
    for channel in channels:
        if channel != "clean":
            all_rows.extend(apply_channel(channel, clean_rows, source_paths, CHANNELED_DIR))
    df = pd.DataFrame(all_rows)
    df.to_csv(MANIFEST_PATH, index=False)
    print(f"\nManifest salvo em {MANIFEST_PATH}: {len(df)} janelas.")
    print(df.groupby(["channel", "generator"]).size())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["prepare", "build"])
    parser.add_argument("--channels", nargs="+", default=ALL_CHANNELS, choices=ALL_CHANNELS)
    args = parser.parse_args()
    if args.step == "prepare":
        prepare()
    else:
        build(args.channels)


if __name__ == "__main__":
    main()
