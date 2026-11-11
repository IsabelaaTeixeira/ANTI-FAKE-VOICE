"""Demo ao vivo pelo microfone: grava janelas de 1s e classifica cada uma
como REAL ou FAKE em tempo real. Dois modelos disponíveis: MFCC + Random
Forest (leve) ou Wav2Vec2 + Regressão Logística (mais pesado e mais preciso;
usa a camada W2V_LAYER de src/features.py, com a rede cortada nessa camada).

Usa os modelos finais do regime augmented, salvos por
`python -m src.train --dataset <dataset> --save-final`.

Usa `arecord` (ALSA) diretamente por subprocess, sem dependências extras
de sistema (nada de portaudio/sounddevice).

Uso:
    python -m src.live_demo --seconds 15
    python -m src.live_demo --seconds 15 --features wav2vec2
    python -m src.live_demo --seconds 15 --features wav2vec2 --dataset asvspoof --int8
    python -m src.live_demo --seconds 15 --features wav2vec2 --model models/wav2vec2_with_own_voice.joblib
"""
import argparse
import subprocess
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np

from src import channel_sim
from src.evaluate import logit
from src.features import SR, W2V_LAYER, extract_mfcc_from_array
from src.train import final_model_path


def load_wav2vec2_extractor(int8: bool):
    from src.latency_bench import wav2vec2_variant

    return wav2vec2_variant(W2V_LAYER, int8)


def extract_wav2vec2_embedding(audio: np.ndarray, processor, model) -> np.ndarray:
    from src.latency_bench import embed

    return embed([audio], processor, model)[0]


def record_one_second(tmp_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Grava 1s e devolve (áudio cru, áudio normalizado). O normalizado passa pela
    MESMA normalização de loudness (-16 LUFS) aplicada a toda janela do treino —
    sem ela, o modelo receberia ao vivo um volume que nunca viu no treino."""
    import soundfile as sf

    subprocess.run(
        ["arecord", "-q", "-d", "1", "-f", "S16_LE", "-r", str(SR), "-c", "1", str(tmp_path)],
        check=True, capture_output=True,
    )
    normalized_path = tmp_path.with_name("chunk_norm.wav")
    channel_sim.normalize_clean(tmp_path, normalized_path)
    raw, _ = sf.read(tmp_path, dtype="float32")
    normalized, _ = sf.read(normalized_path, dtype="float32")
    return raw, normalized


SILENCE_RMS_THRESHOLD = 0.01  # abaixo disso, provavelmente não tem fala de verdade na janela


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=15, help="duração total da demo, em segundos")
    parser.add_argument("--dataset", choices=["deepvoice", "asvspoof"], default="deepvoice")
    parser.add_argument("--features", choices=["mfcc", "wav2vec2"], default="mfcc")
    parser.add_argument("--int8", action="store_true", help="Wav2Vec2 quantizado em int8 (mais rápido)")
    parser.add_argument("--model", type=Path, default=None, help="caminho do modelo .joblib a usar (padrão: o final do --dataset/--features escolhidos)")
    args = parser.parse_args()

    model_path = args.model or final_model_path(args.features, "augmented", args.dataset)
    if not model_path.exists():
        raise SystemExit(
            f"Modelo não encontrado em {model_path}. Gere-o com:\n"
            f"    python -m src.train --dataset {args.dataset} --features {args.features} --save-final"
        )
    clf = joblib.load(model_path)
    print(f"Dataset: {args.dataset} · Atributos: {args.features} · Usando modelo: {model_path}\n")

    w2v_processor = w2v_model = None
    if args.features == "wav2vec2":
        print(f"Carregando o extrator Wav2Vec2 (camada {W2V_LAYER}{', int8' if args.int8 else ''})...")
        w2v_processor, w2v_model = load_wav2vec2_extractor(args.int8)

    print("Preparando a gravação — respire fundo, já vai começar.", flush=True)
    for aviso in ("3...", "2...", "1...", "FALE AGORA!"):
        time.sleep(1)
        print(aviso, flush=True)

    print(f"\nGravando {args.seconds}s pelo microfone.\n")
    print(f"{'segundo':>8} | {'veredito':<8} | prob. de ser clonada | {'acumulado':<16} | nível de áudio")
    print("-" * 86)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "chunk.wav"
        logodds_history = []
        for i in range(1, args.seconds + 1):
            t0 = time.perf_counter()
            raw, audio = record_one_second(tmp_path)
            rms = float(np.sqrt(np.mean(raw.astype(np.float64) ** 2)))

            if args.features == "mfcc":
                feat = extract_mfcc_from_array(audio)
            else:
                feat = extract_wav2vec2_embedding(audio, w2v_processor, w2v_model)
            proba_fake = clf.predict_proba([feat])[0, 1]
            elapsed = (time.perf_counter() - t0) * 1000

            veredito = "FAKE" if proba_fake >= 0.5 else "real"
            barra = "█" * int(proba_fake * 20)
            silent = rms < SILENCE_RMS_THRESHOLD
            nivel = "silêncio?" if silent else "som captado"
            # veredito acumulado da "ligação": média do log-odds dos segundos com som (ver src/call_level.py)
            if not silent:
                logodds_history.append(float(logit(np.array([proba_fake]))[0]))
            if logodds_history:
                proba_acc = 1 / (1 + np.exp(-np.mean(logodds_history)))
                acumulado = f"{'FAKE' if proba_acc >= 0.5 else 'real'} {proba_acc:>5.1%}"
            else:
                acumulado = "—"
            print(f"{i:>7}s | {veredito:<8} | {proba_fake:>5.1%} {barra:<20} | {acumulado:<16} | {nivel} ({elapsed:.0f}ms)")


if __name__ == "__main__":
    main()
