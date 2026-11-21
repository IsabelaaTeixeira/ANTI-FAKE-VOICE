"""Três gráficos complementares recomendados pela leitura dos artigos de
referência (fase U): curva DET (trade-off completo de erro, não só o ponto de
EER), t-SNE dos embeddings (para investigar se o classificador separa por
locutor/canal em vez de aprender sinal genuíno de síntese) e espectrogramas
comparativos entre canais (qualitativo, mostra o que cada simulação de canal
faz com o áudio).

Uso:
    python -m src.plot_metrics --dataset deepvoice
    python -m src.plot_metrics --dataset asvspoof --plots det tsne
"""
import argparse
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import DetCurveDisplay

from src.features import DATASET_PATHS as FEATURE_PATHS, SR, W2V_LAYER, load_mfcc, load_wav2vec2

PRED_PATHS = {
    "deepvoice": Path("results/cv_predictions.csv"),
    "asvspoof": Path("results/cv_predictions_asvspoof.csv"),
}
FIG_DIR = Path("results/figures")
CHANNELS = ["clean", "gsm", "volte", "vonr", "vonr_loss", "mic"]
CHANNELED_DIR = {"deepvoice": Path("data/channeled"), "asvspoof": Path("data/channeled_asvspoof")}


def plot_det(dataset: str, out_dir: Path) -> None:
    df = pd.read_csv(PRED_PATHS[dataset])
    regime = "augmented"
    split_filter = {"split": "eval"} if dataset == "asvspoof" else {}

    for feature in df["feature"].unique():
        fig, ax = plt.subplots(figsize=(6, 6))
        sub = df[(df["feature"] == feature) & (df["regime"] == regime)]
        for key, val in split_filter.items():
            sub = sub[sub[key] == val]
        for channel in CHANNELS:
            ch_df = sub[sub["channel"] == channel]
            if ch_df.empty or ch_df["label"].nunique() < 2:
                continue
            DetCurveDisplay.from_predictions(ch_df["label"], ch_df["proba"], name=channel, ax=ax)
        ax.set_title(f"Curva DET · {dataset} · {feature} · regime {regime}"
                     + (f" · split {split_filter['split']}" if split_filter else ""), fontsize=10)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out_path = out_dir / f"det_{feature}.png"
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f"Salvo {out_path}")


def _load_embeddings(dataset: str, kind: str) -> tuple[np.ndarray, pd.DataFrame]:
    meta, X = load_mfcc(dataset) if kind == "mfcc" else load_wav2vec2(dataset, W2V_LAYER)
    return X, meta.reset_index(drop=True)


def plot_tsne(dataset: str, out_dir: Path, n_sample: int = 3000, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    for kind in ("mfcc", "wav2vec2"):
        X, meta = _load_embeddings(dataset, kind)
        n = len(meta)
        idx = rng.choice(n, size=min(n_sample, n), replace=False)
        X_sub, meta_sub = X[idx], meta.iloc[idx].reset_index(drop=True)

        if X_sub.shape[1] > 50:
            X_sub = PCA(n_components=50, random_state=seed).fit_transform(X_sub)
        emb = TSNE(n_components=2, init="pca", perplexity=30, random_state=seed).fit_transform(X_sub)

        if dataset == "deepvoice":
            color_by = [("canal", meta_sub["channel"]), ("locutor", meta_sub["speakers"])]
        else:
            known = meta_sub["split"].map({"train": "conhecido (treino)", "dev": "desconhecido (dev)", "eval": "desconhecido (eval)"})
            color_by = [("canal", meta_sub["channel"]), ("locutor: conhecido x desconhecido", known)]

        fig, axes = plt.subplots(1, len(color_by), figsize=(8.5 * len(color_by), 6))
        if len(color_by) == 1:
            axes = [axes]
        for ax, (label, series) in zip(axes, color_by):
            cats = series.astype("category")
            codes = cats.cat.codes
            n_cats = len(cats.cat.categories)
            cmap = "tab10" if n_cats <= 10 else "tab20"
            scatter = ax.scatter(emb[:, 0], emb[:, 1], c=codes, cmap=cmap, s=8, alpha=0.7)
            handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=scatter.cmap(scatter.norm(i)))
                       for i in range(n_cats)]
            ax.legend(handles, list(cats.cat.categories), loc="upper left", bbox_to_anchor=(1.02, 1),
                      fontsize=7, markerscale=1.1, ncol=1 if n_cats <= 12 else 2)
            ax.set_title(f"t-SNE {kind}{f' (camada {W2V_LAYER})' if kind == 'wav2vec2' else ''} · colorido por {label}")
            ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        out_path = out_dir / f"tsne_{kind}.png"
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f"Salvo {out_path}")


def plot_spectrograms(dataset: str, out_dir: Path) -> None:
    manifest = pd.read_csv(FEATURE_PATHS[dataset]["manifest"])
    real_label = "real"
    example = manifest[(manifest["label"] == real_label) & (manifest["channel"] == "clean")].iloc[0]
    rows = manifest[
        (manifest["source_file"] == example["source_file"])
        & (manifest["window_index"] == example["window_index"])
        & (manifest["label"] == real_label)
    ]

    fig, axes = plt.subplots(1, len(CHANNELS), figsize=(4 * len(CHANNELS), 4), sharey=True)
    for ax, channel in zip(axes, CHANNELS):
        match = rows[rows["channel"] == channel]
        if match.empty:
            ax.set_visible(False)
            continue
        y, _ = librosa.load(match.iloc[0]["filepath"], sr=SR, mono=True)
        mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=64)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        librosa.display.specshow(mel_db, sr=SR, x_axis="time", y_axis="mel", ax=ax)
        ax.set_title(channel)
    fig.suptitle(f"Espectrogramas comparativos · {dataset} · {example['source_file']} janela {example['window_index']}")
    fig.tight_layout()
    out_path = out_dir / "spectrograms_comparativo.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Salvo {out_path}")


def main() -> None:
    import librosa.display  # noqa: F401 (precisa estar importado antes de specshow)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["deepvoice", "asvspoof"], required=True)
    parser.add_argument("--plots", nargs="+", choices=["det", "tsne", "spectrogram"],
                         default=["det", "tsne", "spectrogram"])
    args = parser.parse_args()

    out_dir = FIG_DIR / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    if "det" in args.plots:
        plot_det(args.dataset, out_dir)
    if "tsne" in args.plots:
        plot_tsne(args.dataset, out_dir)
    if "spectrogram" in args.plots:
        plot_spectrograms(args.dataset, out_dir)


if __name__ == "__main__":
    main()
