"""Demo ao vivo pelo microfone: grava janelas de 1s e classifica cada uma
como REAL ou FAKE em tempo real. Dois modelos disponíveis: MFCC + Random
Forest (leve, o candidato a rodar num nó MEC de verdade, ver etapa 6) ou
Wav2Vec2 + Regressão Logística (mais pesado, não cabe no orçamento de
latência de 150ms da fase L, mas útil para comparar se a augmentation de
canal ajuda mais nele do que no MFCC, como visto na fase Y).

Usa `arecord` (ALSA) diretamente por subprocess, sem dependências extras
de sistema (nada de portaudio/sounddevice).

Uso:
    python -m src.live_demo --seconds 15
    python -m src.live_demo --seconds 15 --features wav2vec2
    python -m src.live_demo --seconds 15 --dataset asvspoof
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

from src.features import META_COLS, MFCC_OUT, SR, extract_mfcc_from_array
from src.train import make_classifier

DEFAULT_MFCC_MODEL_PATH = Path("models/mfcc_rf_final.joblib")
DEFAULT_W2V_MODEL_PATH = Path("models/wav2vec2_lr_final.joblib")

DATASET_PATHS = {
    "deepvoice": {
        "meta_cols": list(META_COLS),
        "mfcc_features": MFCC_OUT,
        "mfcc_model": DEFAULT_MFCC_MODEL_PATH,
        "w2v_array": Path("data/features_wav2vec2.npz"),
        "w2v_meta": Path("data/features_wav2vec2_meta.csv"),
        "w2v_model": DEFAULT_W2V_MODEL_PATH,
    },
    "asvspoof": {
        "meta_cols": list(META_COLS) + ["split", "attack_id"],
        "mfcc_features": Path("data/features_mfcc_asvspoof.csv"),
        "mfcc_model": Path("models/mfcc_rf_final_asvspoof.joblib"),
        "w2v_array": Path("data/features_wav2vec2_asvspoof.npz"),
        "w2v_meta": Path("data/features_wav2vec2_meta_asvspoof.csv"),
        "w2v_model": Path("models/wav2vec2_lr_final_asvspoof.joblib"),
    },
}


def load_or_train_mfcc(model_path: Path, mfcc_path: Path, meta_cols: list[str], default_model_path: Path):
    if model_path.exists():
        return joblib.load(model_path)
    if model_path != default_model_path:
        raise SystemExit(f"Modelo não encontrado em {model_path}.")

    print(f"Treinando o modelo final (MFCC + Random Forest, todos os canais, {mfcc_path})...")
    df = pd.read_csv(mfcc_path)
    feat_cols = [c for c in df.columns if c not in meta_cols]
    clf = make_classifier("mfcc")
    clf.fit(df[feat_cols].values, (df["label"] == "fake").astype(int).values)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"Modelo salvo em {model_path}\n")
    return clf


def load_or_train_wav2vec2(model_path: Path, array_path: Path, meta_path: Path, default_model_path: Path):
    if model_path.exists():
        return joblib.load(model_path)
    if model_path != default_model_path:
        raise SystemExit(f"Modelo não encontrado em {model_path}.")

    print(f"Treinando o modelo final (Wav2Vec2 + Regressão Logística, todos os canais, {array_path})...")
    meta = pd.read_csv(meta_path)
    X = np.load(array_path)["X"]
    y = (meta["label"] == "fake").astype(int).values
    clf = make_classifier("wav2vec2")
    clf.fit(X, y)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"Modelo salvo em {model_path}\n")
    return clf


def load_wav2vec2_extractor():
    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device).eval()
    return processor, model, device


def extract_wav2vec2_embedding(audio: np.ndarray, processor, model, device) -> np.ndarray:
    import torch

    inputs = processor([audio], sampling_rate=SR, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(inputs.input_values.to(device))
    return out.last_hidden_state.mean(dim=1).cpu().numpy()[0]


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
    parser.add_argument("--dataset", choices=["deepvoice", "asvspoof"], default="deepvoice")
    parser.add_argument("--features", choices=["mfcc", "wav2vec2"], default="mfcc")
    parser.add_argument("--model", type=Path, default=None, help="caminho do modelo .joblib a usar (padrão: o final do --dataset/--features escolhidos)")
    args = parser.parse_args()

    paths = DATASET_PATHS[args.dataset]
    w2v_processor = w2v_model = w2v_device = None

    if args.features == "mfcc":
        model_path = args.model or paths["mfcc_model"]
        print(f"Dataset: {args.dataset} · Atributos: MFCC · Usando modelo: {model_path}\n")
        clf = load_or_train_mfcc(model_path, paths["mfcc_features"], paths["meta_cols"], paths["mfcc_model"])
    else:
        model_path = args.model or paths["w2v_model"]
        print(f"Dataset: {args.dataset} · Atributos: Wav2Vec2 · Usando modelo: {model_path}\n")
        clf = load_or_train_wav2vec2(model_path, paths["w2v_array"], paths["w2v_meta"], paths["w2v_model"])
        print("Carregando o extrator Wav2Vec2 (pode demorar um pouco na primeira vez)...")
        w2v_processor, w2v_model, w2v_device = load_wav2vec2_extractor()

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

            if args.features == "mfcc":
                feat = extract_mfcc_from_array(audio)
            else:
                feat = extract_wav2vec2_embedding(audio, w2v_processor, w2v_model, w2v_device)
            proba_fake = clf.predict_proba([feat])[0, 1]
            elapsed = (time.perf_counter() - t0) * 1000

            veredito = "FAKE" if proba_fake >= 0.5 else "real"
            barra = "█" * int(proba_fake * 20)
            nivel = "silêncio?" if rms < SILENCE_RMS_THRESHOLD else "som captado"
            print(f"{i:>7}s | {veredito:<8} | {proba_fake:>5.1%} {barra:<20} | {nivel} ({elapsed:.0f}ms)")


if __name__ == "__main__":
    main()
