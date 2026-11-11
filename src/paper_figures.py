"""Figuras do artigo de congresso (PDF vetorial, largura de uma coluna do modelo IEEE).

  fig_canais.pdf     eval do ASVspoof 2019: EER por canal x regime de treino,
                     para os dois classificadores treinados no split completo
  fig_porjanela.pdf  controle da simulação: codec na gravação inteira x codec em
                     cada janela isolada, modelos treinados só com áudio limpo
  fig_2021.pdf       ASVspoof 2021 LA (transmissão real): EER por gravação em cada
                     condição, comparado aos detectores de referência do desafio

Lê results/full_*.csv (src/full_tables.py) e results/asvspoof2021_baselines.csv.
Uma figura sem dados é pulada com aviso.

Uso:
    python -m src.paper_figures
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path("results")
OUT_DIR = Path("artigo/congresso/figuras")

COLUMN_WIDTH = 3.5  # polegadas, uma coluna do IEEEtran
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
# paleta categórica validada para daltonismo (3 primeiras posições); cada regime tem sempre a mesma cor
REGIME_ORDER = ["clean", "augmented", "augmented_matched"]
REGIME_LABELS = {"clean": "Limpo", "augmented": "Augmentado", "augmented_matched": "Pareado"}
REGIME_LEGEND = {"clean": "Treino limpo", "augmented": "Augmentado (5 canais)", "augmented_matched": "Pareado"}
REGIME_COLORS = {"clean": "#2a78d6", "augmented": "#eb6834", "augmented_matched": "#1baf7a"}
CLEAN_LIGHT = "#86b6ef"  # mesmo azul, tom mais claro (degrau 250 da rampa)
BASELINE_COLORS = {"LFCC-LCNN": INK_SECONDARY, "RawNet2": "#a9a8a2"}
CLASSIFIER_LABELS = {"lr": "Média + RL", "seq": "Sequência"}

CHANNEL_ORDER = ["clean", "gsm", "volte", "vonr", "vonr_loss", "mic"]
CHANNEL_LABELS = {"clean": "Limpo", "gsm": "GSM", "volte": "VoLTE", "vonr": "VoNR",
                  "vonr_loss": "VoNR\n+perda", "mic": "Mic"}
REAL_ORDER = ["none", "alaw", "ulaw", "g722", "gsm", "opus", "pstn", "todas"]
REAL_LABELS = {"none": "Sem\ncodec", "alaw": "a-law", "ulaw": "µ-law", "g722": "G.722",
               "gsm": "GSM", "opus": "Opus", "pstn": "PSTN", "todas": "Todas"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "DejaVu Serif"],
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": INK_SECONDARY, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 0,
    "hatch.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,  # fontes TrueType embutidas (exigência do IEEE PDF eXpress)
})


def style_axis(ax, ymax: float) -> None:
    ax.set_ylim(0, ymax)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


def grouped_bars(ax, groups: list[str], series: list[dict], width: float = 0.8) -> None:
    """series: dicts com values, low, high (opcionais), color, label, hatch."""
    n = len(series)
    bar = width / n
    x = np.arange(len(groups))
    for k, s in enumerate(series):
        pos = x - width / 2 + bar * (k + 0.5)
        yerr = None
        if s.get("low") is not None:
            values = np.asarray(s["values"])
            yerr = np.vstack([values - np.asarray(s["low"]), np.asarray(s["high"]) - values])
        ax.bar(pos, s["values"], bar * 0.9, color=s["color"], label=s["label"], hatch=s.get("hatch"),
               edgecolor=s.get("edge", "none"), linewidth=0,
               yerr=yerr, error_kw={"elinewidth": 0.5, "capsize": 0, "ecolor": INK_SECONDARY})
    ax.set_xticks(x)
    ax.set_xticklabels(groups)


def save(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / name, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Salvo {OUT_DIR / name}")


def fig_canais() -> None:
    t = pd.read_csv(RESULTS / "full_asvspoof2019.csv")
    t = t[(t["split"] == "eval") & (t["train_set"] == "full")]
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 3.3), sharex=True)
    ymax = np.ceil(t["eer_ci_high"].max() * 20) / 20
    for ax, classifier in zip(axes, ["lr", "seq"]):
        sub = t[t["classifier"] == classifier].set_index(["regime", "channel"])
        series = [{
            "values": [sub.loc[(r, c), "eer"] for c in CHANNEL_ORDER],
            "low": [sub.loc[(r, c), "eer_ci_low"] for c in CHANNEL_ORDER],
            "high": [sub.loc[(r, c), "eer_ci_high"] for c in CHANNEL_ORDER],
            "color": REGIME_COLORS[r], "label": REGIME_LEGEND[r],
        } for r in REGIME_ORDER]
        grouped_bars(ax, [CHANNEL_LABELS[c] for c in CHANNEL_ORDER], series)
        style_axis(ax, ymax)
        ax.set_ylabel("EER por janela")
        ax.set_title(CLASSIFIER_LABELS[classifier], loc="left", fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               handlelength=1.2, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=0.6)
    save(fig, "fig_canais.pdf")


def fig_porjanela() -> None:
    t = pd.read_csv(RESULTS / "full_perwindow.csv")
    t = t[(t["windows"] == "a partir da 2a") & (t["train_set"] == "full") & (t["regime"] == "clean")]
    channels = ["gsm", "volte", "vonr"]
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_WIDTH, 1.75), sharey=True)
    ymax = np.ceil(t["eer_ci_high"].max() * 20) / 20
    for ax, classifier in zip(axes, ["lr", "seq"]):
        sub = t[t["classifier"] == classifier].set_index(["simulation", "channel"])
        series = [
            {"values": [sub.loc[("continuous", c), "eer"] for c in channels],
             "low": [sub.loc[("continuous", c), "eer_ci_low"] for c in channels],
             "high": [sub.loc[("continuous", c), "eer_ci_high"] for c in channels],
             "color": REGIME_COLORS["clean"], "label": "Codec na gravação inteira"},
            {"values": [sub.loc[("per_window", c), "eer"] for c in channels],
             "low": [sub.loc[("per_window", c), "eer_ci_low"] for c in channels],
             "high": [sub.loc[("per_window", c), "eer_ci_high"] for c in channels],
             "color": CLEAN_LIGHT, "hatch": "////", "edge": REGIME_COLORS["clean"], "label": "Codec em cada janela"},
        ]
        grouped_bars(ax, [CHANNEL_LABELS[c] for c in channels], series, width=0.7)
        clean = sub.loc[("continuous", "clean"), "eer"]
        ax.axhline(clean, color=INK_SECONDARY, linewidth=0.7, linestyle=(0, (3, 2)), label="Sem codec")
        style_axis(ax, ymax)
        ax.set_title(CLASSIFIER_LABELS[classifier], loc="left", fontweight="bold")
    axes[0].set_ylabel("EER por janela")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               handlelength=1.4, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=0.8)
    save(fig, "fig_porjanela.pdf")


def fig_2021(models: list[str]) -> None:
    ours = pd.read_csv(RESULTS / "full_asvspoof2021.csv")
    ours = ours[ours["generator"] == "todos"].set_index(["model", "channel"])
    base = pd.read_csv(RESULTS / "asvspoof2021_baselines.csv").set_index(["baseline", "channel"])
    series = []
    for m in models:
        classifier, _, regime = m.split("_", 2)
        series.append({"values": [ours.loc[(m, c), "eer_utterance"] for c in REAL_ORDER],
                       "color": REGIME_COLORS[regime],
                       "label": f"{CLASSIFIER_LABELS[classifier]}, treino {REGIME_LABELS[regime].lower()}"})
    for name, color in BASELINE_COLORS.items():
        series.append({"values": [base.loc[(name, c), "eer_utterance_with_windows"] for c in REAL_ORDER],
                       "color": color, "label": name})
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 1.95))
    grouped_bars(ax, [REAL_LABELS[c] for c in REAL_ORDER], series, width=0.82)
    ax.axvline(len(REAL_ORDER) - 1.5, color=GRID, linewidth=0.8)
    top = max(max(s["values"]) for s in series)
    style_axis(ax, np.ceil(top * 1.05 * 20) / 20)
    ax.set_ylabel("EER por gravação")
    ax.legend(ncol=2, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), handlelength=1.2, columnspacing=1.2)
    fig.tight_layout()
    save(fig, "fig_2021.pdf")


def main(models_2021: list[str] | None = None) -> None:
    models_2021 = models_2021 or ["seq_full_clean", "seq_full_augmented"]
    for build in (fig_canais, fig_porjanela, lambda: fig_2021(models_2021)):
        try:
            build()
        except (FileNotFoundError, KeyError) as error:
            print(f"Figura pulada (dados ausentes): {error!r}")


if __name__ == "__main__":
    main()
