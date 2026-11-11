"""Ideia 2 da usuária: retreina o classificador incluindo a própria voz (real +
clonada via XTTS) e testa numa parte dessa voz que NÃO entrou no retreino.

Pipeline, igual ao resto do projeto (build_dataset.slice_to_windows): remove
o silêncio de borda, passa a gravação inteira por cada canal, fatia em
janelas de 1s normalizadas a -16 LUFS, e extrai os mesmos atributos.

Gravações usadas (cada pessoa grava a própria voz, ver README):
  - voz real: data/my_voice_reference*.wav (uma ou mais sessões)
  - voz clonada: data/cloned_voice*.wav

Como separar treino e teste:
  - com 2+ gravações reais (sessões diferentes): a ÚLTIMA sessão real e a
    última clonada ficam só para teste — o teste certo, porque uma sessão nova
    muda microfone, sala e jeito de falar, como no uso real;
  - com uma gravação só: divide cada gravação no tempo (70% iniciais no
    treino, 5% descartados, 25% finais no teste), como em
    speaker_dependent_eval.py. Com ~15s de voz real isso deixa só ~3 janelas
    de teste por canal: o número é ilustrativo, não estatístico.

Também salva um modelo retreinado com TODA a voz própria
(models/<atributo>_with_own_voice.joblib), para testar ao vivo numa gravação
nova com src/live_demo.py --model.

Uso:
    python -m src.own_voice_ceiling [--features mfcc|wav2vec2|both]
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from src import mic_sim
from src.build_dataset import ALL_CHANNELS, _file_seed, slice_to_windows
from src.evaluate import eer
from src.features import (
    SR, W2V_LAYER, extract_mfcc, load_mfcc, load_wav2vec2, load_wav2vec2_model,
    wav2vec2_layer_embeddings,
)
from src.speaker_dependent_eval import GAP_FRACTION, TRAIN_FRACTION
from src.train import label_vector, make_classifier, regime_mask

REAL_GLOB = "my_voice_reference*.wav"
FAKE_GLOB = "cloned_voice*.wav"
DATA_DIR = Path("data")
WINDOWS_DIR = Path("data/raw/windows_own_voice")
SPEAKER_TAG = "propria_voz"
OUT_PATH = Path("results/own_voice_ceiling_table.csv")


def own_recordings() -> dict[str, list[Path]]:
    recordings = {"real": sorted(DATA_DIR.glob(REAL_GLOB)), "fake": sorted(DATA_DIR.glob(FAKE_GLOB))}
    if not recordings["real"] or not recordings["fake"]:
        raise SystemExit("Grave data/my_voice_reference.wav e gere data/cloned_voice.wav antes (ver README).")
    return recordings


def build_own_voice_rows() -> pd.DataFrame:
    """Janelas da própria voz em todos os canais, no mesmo formato do manifest."""
    rows = []
    for label, paths in own_recordings().items():
        for session, src in enumerate(paths):
            clean = slice_to_windows(src, WINDOWS_DIR / "clean" / label)
            for channel in ALL_CHANNELS:
                if channel == "clean":
                    windows = clean
                elif channel == "mic":
                    windows = []
                    for clean_path in clean:
                        out = WINDOWS_DIR / "mic" / label / clean_path.name
                        out.parent.mkdir(parents=True, exist_ok=True)
                        if not out.exists():
                            mic_sim.to_mic(clean_path, out, seed=_file_seed(clean_path))
                        windows.append(out)
                else:
                    windows = slice_to_windows(src, WINDOWS_DIR / channel / label, channel)
                for idx, path in enumerate(windows):
                    rows.append({
                        "filepath": str(path), "speakers": SPEAKER_TAG, "label": label, "channel": channel,
                        "source_file": src.name, "window_index": idx, "session": session,
                        "n_sessions": len(paths),
                    })
    return pd.DataFrame(rows)


def own_voice_features(own_df: pd.DataFrame, feature_name: str) -> np.ndarray:
    if feature_name == "mfcc":
        return np.array([extract_mfcc(p) for p in own_df["filepath"]])
    import librosa

    processor, model = load_wav2vec2_model()
    audios = [librosa.load(p, sr=SR, mono=True)[0] for p in own_df["filepath"]]
    return np.concatenate([
        wav2vec2_layer_embeddings(audios[i:i + 16], processor, model)[W2V_LAYER]
        for i in range(0, len(audios), 16)
    ])


def own_voice_split(own_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, str]:
    if (own_df["n_sessions"] >= 2).all():
        is_test = (own_df["session"] == own_df["n_sessions"] - 1).values
        return ~is_test, is_test, "sessão nova (última gravação fora do treino)"
    n_windows = own_df.groupby(["source_file", "channel"])["window_index"].transform("max") + 1
    position = own_df["window_index"] / n_windows
    return (
        (position < TRAIN_FRACTION).values,
        (position >= TRAIN_FRACTION + GAP_FRACTION).values,
        "mesma gravação, trecho final (25%) fora do treino",
    )


def summarize(own_df: pd.DataFrame, proba: np.ndarray, **fixed) -> list[dict]:
    rows = []
    for channel in own_df["channel"].unique():
        mask = (own_df["channel"] == channel).values
        y = label_vector(own_df[mask])
        pred = (proba[mask] >= 0.5).astype(int)
        rows.append({
            **fixed, "channel": channel, "n": int(mask.sum()), "n_real": int((y == 0).sum()),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "real_correct": float((pred[y == 0] == 0).mean()), "fake_correct": float((pred[y == 1] == 1).mean()),
            "eer": eer(y, proba[mask]),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", choices=["mfcc", "wav2vec2", "both"], default="both")
    args = parser.parse_args()

    print("Fatiando e aplicando os canais na sua voz (real + clonada)...")
    own_df = build_own_voice_rows()
    is_train, is_test, split_desc = own_voice_split(own_df)
    print(f"{len(own_df)} janelas da própria voz; teste: {split_desc}.\n")

    rows = []
    for feature_name in ("mfcc", "wav2vec2"):
        if args.features not in (feature_name, "both"):
            continue
        base_df, X_base = load_mfcc("deepvoice") if feature_name == "mfcc" else load_wav2vec2("deepvoice")
        X_own = own_voice_features(own_df, feature_name)
        base_mask = regime_mask("augmented", base_df["channel"].values, np.ones(len(base_df), dtype=bool))
        own_mask = regime_mask("augmented", own_df["channel"].values, np.ones(len(own_df), dtype=bool))
        y_base, y_own = label_vector(base_df), label_vector(own_df)

        print(f"[{feature_name}] Retreinando com a parte de treino da voz própria e testando no resto...")
        clf = make_classifier(feature_name).fit(
            np.vstack([X_base[base_mask], X_own[own_mask & is_train]]),
            np.concatenate([y_base[base_mask], y_own[own_mask & is_train]]),
        )
        rows += summarize(own_df[is_test], clf.predict_proba(X_own[is_test])[:, 1], feature=feature_name, test=split_desc)

        print(f"[{feature_name}] Salvando modelo retreinado com TODA a voz própria (para teste ao vivo)...")
        clf_all = make_classifier(feature_name).fit(
            np.vstack([X_base[base_mask], X_own[own_mask]]),
            np.concatenate([y_base[base_mask], y_own[own_mask]]),
        )
        out_model = Path(f"models/{feature_name}_with_own_voice.joblib")
        joblib.dump(clf_all, out_model)
        print(f"Modelo salvo em {out_model}\n")

    summary = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(summary.to_string(index=False))
    summary.to_csv(OUT_PATH, index=False)
    print(f"\nTabela salva em {OUT_PATH}")


if __name__ == "__main__":
    main()
