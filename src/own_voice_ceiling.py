"""Ideia 2 da usuária (versão rápida, com o áudio que já temos): retreina
o classificador incluindo a própria voz (real + clonada via XTTS) e testa
nela mesma — a MESMA metodologia usada em speaker_dependent_eval.py para
Taylor/Biden/Ryan, pra dar um número comparável (o "teto" de desempenho).

Pipeline, igual ao resto do projeto:
  1. Fatia data/my_voice_reference.wav (real) e data/cloned_voice.wav (fake)
     em janelas de 1s, 16kHz mono.
  2. Aplica os mesmos 4 canais (clean/gsm/volte/vonr) via channel_sim.py.
  3. Extrai os mesmos 26 atributos MFCC.
  4. Junta com data/features_mfcc.csv e retreina um RandomForestClassifier
     com os mesmos hiperparâmetros de models/mfcc_rf_final.joblib.
  5. Avalia SÓ nas linhas da própria voz (estavam no treino — mesma lógica
     de speaker_dependent_eval.py), imprimindo janela a janela.

RESSALVA METODOLÓGICA: a voz própria tem muito menos dados (~15s real +
~9s clonada = ~24 janelas antes de aplicar canal) do que qualquer um dos
6 locutores do dataset original (centenas a milhares de janelas cada).
Isso torna a comparação com a tabela de Taylor/Biden/Ryan ilustrativa,
não estatisticamente equivalente.

Uso:
    python -m src.own_voice_ceiling
"""
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_curve

from src import channel_sim
from src.features import META_COLS, MFCC_OUT, SR, extract_mfcc_from_array

REAL_SRC = Path("data/my_voice_reference.wav")
FAKE_SRC = Path("data/cloned_voice.wav")
WINDOWS_DIR = Path("data/raw/windows_own_voice")
SPEAKER_TAG = "propria_voz"
CHANNELS = ("clean", "gsm", "volte", "vonr")
CHANNEL_FN = {"gsm": channel_sim.to_gsm, "volte": channel_sim.to_volte, "vonr": channel_sim.to_vonr}


def slice_clean(src: Path, label: str) -> list[Path]:
    out_dir = WINDOWS_DIR / "clean" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    y, _ = librosa.load(src, sr=SR, mono=True)
    n_windows = len(y) // SR
    paths = []
    for i in range(n_windows):
        out_path = out_dir / f"{src.stem}_{i:03d}.wav"
        if not out_path.exists():
            sf.write(out_path, y[i * SR:(i + 1) * SR], SR)
        paths.append(out_path)
    return paths


def build_own_voice_rows() -> pd.DataFrame:
    clean_real = slice_clean(REAL_SRC, "real")
    clean_fake = slice_clean(FAKE_SRC, "fake")

    rows = []
    for label, clean_paths in (("real", clean_real), ("fake", clean_fake)):
        for clean_path in clean_paths:
            for channel in CHANNELS:
                if channel == "clean":
                    window_path = clean_path
                else:
                    out_dir = WINDOWS_DIR / channel / label
                    out_dir.mkdir(parents=True, exist_ok=True)
                    window_path = out_dir / clean_path.name
                    if not window_path.exists():
                        CHANNEL_FN[channel](clean_path, window_path)

                y, _ = librosa.load(window_path, sr=SR, mono=True)
                feat = extract_mfcc_from_array(y)
                rows.append({
                    "filepath": str(window_path),
                    "speakers": SPEAKER_TAG,
                    "label": label,
                    "channel": channel,
                    **dict(zip(_mfcc_columns(), feat)),
                })
    return pd.DataFrame(rows)


def _mfcc_columns() -> list[str]:
    df = pd.read_csv(MFCC_OUT, nrows=1)
    return [c for c in df.columns if c not in META_COLS]


def eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2)


def main() -> None:
    print("Fatiando e aplicando os 4 canais na sua voz (real + clonada)...")
    own_df = build_own_voice_rows()
    print(f"{len(own_df)} janelas da própria voz geradas ({len(own_df)//4} por canal x 4 canais).\n")

    base_df = pd.read_csv(MFCC_OUT)
    feat_cols = [c for c in base_df.columns if c not in META_COLS]
    combined = pd.concat([base_df, own_df[base_df.columns]], ignore_index=True)

    print("Retreinando o Random Forest com a própria voz incluída...")
    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(combined[feat_cols].values, (combined["label"] == "fake").astype(int).values)
    joblib.dump(clf, "models/mfcc_rf_with_own_voice.joblib")
    print("Modelo salvo em models/mfcc_rf_with_own_voice.joblib\n")

    print(f"{'janela':>8} | {'canal':<6} | {'rótulo real':<12} | {'veredito':<8} | prob. de ser clonada")
    print("-" * 66)
    own_eval = own_df.sort_values(["label", "channel", "filepath"]).reset_index(drop=True)
    proba_all = clf.predict_proba(own_eval[feat_cols].values)[:, 1]
    for i, (row, proba) in enumerate(zip(own_eval.itertuples(), proba_all), 1):
        veredito = "FAKE" if proba >= 0.5 else "real"
        barra = "█" * int(proba * 20)
        print(f"{i:>7}  | {row.channel:<6} | {row.label:<12} | {veredito:<8} | {proba:>5.1%} {barra}")

    print("\nResumo por canal:")
    rows = []
    for channel, sub in own_eval.groupby("channel"):
        y = (sub["label"] == "fake").astype(int).values
        proba = clf.predict_proba(sub[feat_cols].values)[:, 1]
        pred = (proba >= 0.5).astype(int)
        rows.append({
            "channel": channel, "n": len(sub),
            "accuracy": (pred == y).mean(),
            "f1": f1_score(y, pred, pos_label=1, zero_division=0),
            "eer": eer(y, proba),
        })
    summary = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(summary.to_string(index=False))
    summary.to_csv("results/own_voice_ceiling_table.csv", index=False)
    print("\nTabela salva em results/own_voice_ceiling_table.csv")


if __name__ == "__main__":
    main()
