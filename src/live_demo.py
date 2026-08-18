"""Demo ao vivo pelo microfone: grava janelas de 1s e classifica cada uma
como REAL ou FAKE em tempo real, usando o modelo leve (MFCC + Random
Forest, regime augmented — o candidato a rodar num nó MEC de verdade,
ver etapa 6).

Usa `arecord` (ALSA) diretamente por subprocess, sem dependências extras
de sistema (nada de portaudio/sounddevice).

Uso:
    python -m src.live_demo --seconds 15
    python -m src.live_demo --seconds 15 --model models/mfcc_rf_with_own_voice.joblib
"""
import argparse
import subprocess
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.features import META_COLS, MFCC_OUT, SR, extract_mfcc_from_array

DEFAULT_MODEL_PATH = Path("models/mfcc_rf_final.joblib")


def load_or_train_model(model_path: Path) -> RandomForestClassifier:
    if model_path.exists():
        return joblib.load(model_path)

    if model_path != DEFAULT_MODEL_PATH:
        raise SystemExit(f"Modelo não encontrado em {model_path}.")

    print("Treinando o modelo final (MFCC + Random Forest, todos os canais)...")
    df = pd.read_csv(MFCC_OUT)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(df[feat_cols].values, (df["label"] == "fake").astype(int).values)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"Modelo salvo em {model_path}\n")
    return clf


def record_one_second(tmp_path: Path) -> np.ndarray:
    subprocess.run(
        ["arecord", "-q", "-d", "1", "-f", "S16_LE", "-r", str(SR), "-c", "1", str(tmp_path)],
        check=True, capture_output=True,
    )
    import soundfile as sf
    y, _ = sf.read(tmp_path, dtype="float32")
    return y


SILENCE_RMS_THRESHOLD = 0.01  # abaixo disso, provavelmente não tem fala de verdade na janela


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=15, help="duração total da demo, em segundos")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="caminho do modelo .joblib a usar")
    args = parser.parse_args()

    print(f"Usando modelo: {args.model}\n")
    clf = load_or_train_model(args.model)

    print("Preparando a gravação — respire fundo, já vai começar.", flush=True)
    for aviso in ("3...", "2...", "1...", "FALE AGORA!"):
        time.sleep(1)
        print(aviso, flush=True)

    print(f"\nGravando {args.seconds}s pelo microfone.\n")
    print(f"{'segundo':>8} | {'veredito':<8} | prob. de ser clonada | nível de áudio")
    print("-" * 66)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "chunk.wav"
        for i in range(1, args.seconds + 1):
            t0 = time.perf_counter()
            audio = record_one_second(tmp_path)
            rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
            feat = extract_mfcc_from_array(audio)
            proba_fake = clf.predict_proba([feat])[0, 1]
            elapsed = (time.perf_counter() - t0) * 1000

            veredito = "FAKE" if proba_fake >= 0.5 else "real"
            barra = "█" * int(proba_fake * 20)
            nivel = "silêncio?" if rms < SILENCE_RMS_THRESHOLD else "som captado"
            print(f"{i:>7}s | {veredito:<8} | {proba_fake:>5.1%} {barra:<20} | {nivel} ({elapsed:.0f}ms)")


if __name__ == "__main__":
    main()
