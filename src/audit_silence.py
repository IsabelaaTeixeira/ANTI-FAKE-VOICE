"""Audita se REAL e FAKE têm padrões de silêncio sistematicamente diferentes
no DEEP-VOICE, inspirado em Müller et al. (2021, "Speech is Silver, Silence
is Golden"), que mostrou que classificadores no ASVspoof aprendiam a duração
do silêncio de borda em vez de conteúdo de fala. Se o mesmo viés existir aqui,
o classificador pode estar "trapaceando" da mesma forma.

Duas checagens:
1. Nível de arquivo original: duração de silêncio de abertura/fechamento em
   cada gravação bruta (antes do fatiamento em janelas de 1s), por rótulo.
2. Nível de janela: RMS médio de cada janela de 1s já usada no treino, por
   rótulo — é a unidade que o classificador realmente vê.

Uso:
    python -m src.audit_silence
"""
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from scipy import stats

RAW_DIR = Path("data/raw/KAGGLE/AUDIO")
WINDOWS_DIR = Path("data/raw/windows")
MANIFEST_PATH = Path("data/manifest.csv")
OUT_DIR = Path("results")

SR = 16000
TOP_DB = 30  # limiar de silêncio (librosa.effects.trim), mesma ordem de grandeza usada por Müller et al.


def edge_silence_seconds(audio_path: Path) -> tuple[float, float]:
    audio, _ = librosa.load(audio_path, sr=SR, mono=True)
    _, index = librosa.effects.trim(audio, top_db=TOP_DB)
    leading = index[0] / SR
    trailing = (len(audio) - index[1]) / SR
    return leading, trailing


def audit_source_files() -> pd.DataFrame:
    rows = []
    for label, subdir in (("real", "REAL"), ("fake", "FAKE")):
        for audio_path in sorted((RAW_DIR / subdir).glob("*.wav")):
            leading, trailing = edge_silence_seconds(audio_path)
            rows.append({
                "source_file": audio_path.name,
                "label": label,
                "leading_silence_s": leading,
                "trailing_silence_s": trailing,
                "total_edge_silence_s": leading + trailing,
            })
    return pd.DataFrame(rows)


def audit_windows() -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST_PATH)
    clean = manifest[manifest["channel"] == "clean"]
    rows = []
    for row in clean.itertuples():
        audio, _ = librosa.load(row.filepath, sr=SR, mono=True)
        rms = float(np.sqrt(np.mean(audio ** 2)))
        rows.append({"filepath": row.filepath, "label": row.label, "rms": rms})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, value_col: str, label_col: str = "label") -> None:
    real = df.loc[df[label_col] == "real", value_col]
    fake = df.loc[df[label_col] == "fake", value_col]
    print(df.groupby(label_col)[value_col].describe())
    u_stat, p_value = stats.mannwhitneyu(real, fake, alternative="two-sided")
    print(f"Mann-Whitney U: estatística={u_stat:.1f}, p-valor={p_value:.4f}")
    if p_value < 0.05:
        print("=> diferença estatisticamente significativa entre REAL e FAKE (possível atalho).")
    else:
        print("=> sem diferença estatisticamente significativa entre REAL e FAKE.")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    print("=== Silêncio de borda por arquivo original (estilo Müller et al. 2021) ===")
    files_df = audit_source_files()
    files_df.to_csv(OUT_DIR / "silence_audit_files.csv", index=False)
    summarize(files_df, "total_edge_silence_s")

    print("\n=== RMS por janela de 1s (unidade usada no treino) ===")
    windows_df = audit_windows()
    windows_df.to_csv(OUT_DIR / "silence_audit_windows_rms.csv", index=False)
    summarize(windows_df, "rms")


if __name__ == "__main__":
    main()
