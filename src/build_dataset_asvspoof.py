"""Monta o manifest do ASVspoof 2019 LA (data/manifest_asvspoof.csv), no mesmo
formato usado pelo DEEP-VOICE (build_dataset.py), para reaproveitar o resto do
pipeline (features.py, train.py, evaluate.py, com --dataset asvspoof) sem
alterações estruturais.

Diferenças em relação ao DEEP-VOICE:
  - O ASVspoof já separa locutores por split (treino/validação/teste não
    repetem ninguém) e o split de teste já usa 13 tipos de ataque nunca
    vistos no treino/validação — por isso train.py, no modo ASVspoof, não
    reproduz o Leave-One-Speaker-Out manual: treina uma vez por regime sobre
    o split de treino e testa nos splits de validação e teste.
  - O corpus inteiro tem ~122 mil gravações; fatiar e aplicar os 4 canais em
    tudo passaria de 1 milhão de janelas (inviável em disco/tempo aqui). Por
    isso a amostragem por grupo (locutor × tipo de ataque/bonafide), com um
    teto por grupo — preserva os 107 locutores e os 13 ataques desconhecidos,
    só limita quantas gravações de cada combinação entram.

Uso:
    python -m src.build_dataset_asvspoof --channels clean
    python -m src.build_dataset_asvspoof --channels clean gsm volte vonr --cap 20
"""
import argparse
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.build_dataset import CHANNEL_FN, slice_to_windows

RAW_DIR = Path("data/raw/ASVSPOOF2019/LA")
PROTOCOLS_DIR = RAW_DIR / "ASVspoof2019_LA_cm_protocols"
WINDOWS_DIR = Path("data/raw/windows_asvspoof")
CHANNELED_DIR = Path("data/channeled_asvspoof")
MANIFEST_PATH = Path("data/manifest_asvspoof.csv")

SPLIT_DIRS = {
    "train": "ASVspoof2019_LA_train",
    "dev": "ASVspoof2019_LA_dev",
    "eval": "ASVspoof2019_LA_eval",
}
PROTOCOL_FILES = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
}
RANDOM_SEED = 42


def load_protocol(split: str) -> pd.DataFrame:
    path = PROTOCOLS_DIR / PROTOCOL_FILES[split]
    df = pd.read_csv(
        path, sep=" ", header=None,
        names=["speaker", "filename", "_unused", "attack_id", "key"],
    )
    df["split"] = split
    df["label"] = df["key"].map({"bonafide": "real", "spoof": "fake"})
    df["attack_id"] = df["attack_id"].replace("-", "bonafide")
    df["group"] = df["speaker"] + "|" + df["attack_id"]
    return df


def sample_by_group(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    rng = random.Random(RANDOM_SEED)
    parts = []
    for _, g in df.groupby("group"):
        idx = list(g.index)
        rng.shuffle(idx)
        parts.append(g.loc[idx[:cap]])
    return pd.concat(parts, ignore_index=True)


def build_clean_windows(cap: int) -> list[dict]:
    rows = []
    for split in ("train", "dev", "eval"):
        protocol = sample_by_group(load_protocol(split), cap)
        flac_dir = RAW_DIR / SPLIT_DIRS[split] / "flac"
        out_dir = WINDOWS_DIR / split
        for row in tqdm(protocol.itertuples(), total=len(protocol), desc=f"Fatiando {split}"):
            audio_path = flac_dir / f"{row.filename}.flac"
            windows = slice_to_windows(audio_path, out_dir / row.label)
            for idx, window_path in enumerate(windows):
                rows.append({
                    "filepath": str(window_path),
                    "speakers": row.speaker,
                    "label": row.label,
                    "channel": "clean",
                    "source_file": audio_path.name,
                    "window_index": idx,
                    "split": split,
                    "attack_id": row.attack_id,
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
    parser.add_argument(
        "--cap", type=int, default=20,
        help="máximo de gravações por grupo (locutor x tipo de ataque/bonafide)",
    )
    args = parser.parse_args()

    clean_rows = build_clean_windows(args.cap)
    all_rows = list(clean_rows) if "clean" in args.channels else []

    for channel in args.channels:
        if channel == "clean":
            continue
        all_rows.extend(apply_channel(channel, clean_rows))

    df = pd.DataFrame(all_rows)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST_PATH, index=False)
    print(f"\nManifest salvo em {MANIFEST_PATH}: {len(df)} janelas.")
    print(df.groupby(["split", "channel", "label"]).size())


if __name__ == "__main__":
    main()
