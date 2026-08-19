"""Gera o diagrama do desenho experimental (pipeline completo), para abrir a
seção de Metodologia do artigo. Diferente de make_figures.py, não lê nenhum
resultado — é só a representação visual das etapas do pipeline.

Uso:
    python -m src.make_diagram
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIG_DIR = Path("results/figures")

COLOR_MAIN = "#cfe2f3"
COLOR_CHANNEL = "#d9ead3"
COLOR_FEATURE = "#fff2cc"
COLOR_RESULT = "#b6d7a8"
COLOR_EXTRA = "#f4cccc"
EDGE = "#434343"


def _box(ax, x, y, w, h, text, color, fontsize=9.5):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        linewidth=1.2, edgecolor=EDGE, facecolor=color, zorder=2,
    ))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, zorder=3, linespacing=1.35)
    return (x, y, w, h)


def _arrow(ax, start, end, **kwargs):
    style = dict(arrowstyle="-|>", lw=1.3, color=EDGE, shrinkA=0, shrinkB=0)
    style.update(kwargs)
    ax.annotate("", xy=end, xytext=start, arrowprops=style, zorder=1)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 16))
    ax.set_xlim(0, 12)
    ax.set_ylim(-1.5, 20.5)
    ax.axis("off")

    # 1. Dataset
    b1 = _box(ax, 6, 19.4, 7, 1.3, "DEEP-VOICE\nREAL + FAKE, 6 locutores", COLOR_MAIN)
    # 2. Fatiamento
    b3 = _box(ax, 6, 17.2, 7, 1.3, "Fatiamento em janelas de 1s\n16kHz, mono", COLOR_MAIN)
    _arrow(ax, (b1[0], b1[1] - b1[3] / 2), (b3[0], b3[1] + b3[3] / 2))

    # 4. Canais (4 boxes lado a lado)
    channel_y = 14.8
    channel_defs = [
        ("Limpo", 1.6), ("GSM\n(AMR-NB 12,2kb/s)", 4.5), ("VoLTE\n(AMR-WB 12,65kb/s)", 7.5), ("VoNR\n(EVS 13,2kb/s)", 10.4),
    ]
    channel_boxes = [_box(ax, x, channel_y, 2.5, 1.5, label, COLOR_CHANNEL, fontsize=8.5) for label, x in channel_defs]
    for cb in channel_boxes:
        _arrow(ax, (6, b3[1] - b3[3] / 2), (cb[0], cb[1] + cb[3] / 2))

    # barra de convergência
    bus_y = 13.1
    ax.plot([1.6, 10.4], [bus_y, bus_y], color=EDGE, lw=1.3, zorder=1)
    for cb in channel_boxes:
        _arrow(ax, (cb[0], cb[1] - cb[3] / 2), (cb[0], bus_y), color=EDGE)

    # 5. Extração de atributos (2 boxes)
    feature_y = 11.4
    f1 = _box(ax, 4, feature_y, 5, 1.4, "MFCC\n26 atributos espectrais\n(Random Forest)", COLOR_FEATURE, fontsize=8.8)
    f2 = _box(ax, 8.6, feature_y, 5, 1.4, "Wav2Vec2\nembedding 768-dim pré-treinado\n(Regressão Logística)", COLOR_FEATURE, fontsize=8.8)
    _arrow(ax, (4, bus_y), (f1[0], f1[1] + f1[3] / 2))
    _arrow(ax, (8.6, bus_y), (f2[0], f2[1] + f2[3] / 2))

    # 6. Split LOSO
    loso_y = 9.2
    b6 = _box(ax, 6, loso_y, 8, 1.3, "Split Leave-One-Speaker-Out (LOSO)\num locutor de cada vez fica de fora do treino", COLOR_MAIN)
    _arrow(ax, (4, f1[1] - f1[3] / 2), (5, b6[1] + b6[3] / 2))
    _arrow(ax, (8.6, f2[1] - f2[3] / 2), (7, b6[1] + b6[3] / 2))

    # 7. Regimes de treino
    train_y = 7.0
    t1 = _box(ax, 3.3, train_y, 5.4, 1.4, "Regime \"limpo\"\ntreina só com canal limpo", COLOR_MAIN, fontsize=8.8)
    t2 = _box(ax, 8.7, train_y, 5.4, 1.4, "Regime \"augmentado\"\ntreina com os 4 canais juntos", COLOR_MAIN, fontsize=8.8)
    _arrow(ax, (4.5, b6[1] - b6[3] / 2), (t1[0], t1[1] + t1[3] / 2))
    _arrow(ax, (7.5, b6[1] - b6[3] / 2), (t2[0], t2[1] + t2[3] / 2))

    # 8. Avaliação
    eval_y = 4.6
    b8 = _box(ax, 6, eval_y, 8.5, 1.4, "Avaliação: Acurácia, F1 e EER\npor canal, testando sempre no locutor de fora (Resultados)", COLOR_RESULT, fontsize=9)
    _arrow(ax, (t1[0], t1[1] - t1[3] / 2), (4.8, b8[1] + b8[3] / 2))
    _arrow(ax, (t2[0], t2[1] - t2[3] / 2), (7.2, b8[1] + b8[3] / 2))

    # Extensões exploratórias
    ax.text(
        6, 2.6, "Extensões exploratórias (ideias adicionadas durante o projeto)",
        ha="center", fontsize=9.5, style="italic", color="#555", zorder=4,
        bbox=dict(facecolor="white", edgecolor="none", pad=3, alpha=0.95),
    )
    ext_y = 0.4
    e1 = _box(ax, 1.9, ext_y, 3.6, 1.6, "Locutor conhecido\n(Taylor/Biden/Ryan avaliados\nin-sample — \"teto\")", COLOR_EXTRA, fontsize=8.2)
    e2 = _box(ax, 6, ext_y, 3.6, 1.6, "Voz própria + XTTS v2\nantes x depois do retreino,\nin-sample", COLOR_EXTRA, fontsize=8.2)
    e3 = _box(ax, 10.1, ext_y, 3.6, 1.6, "Teste ao vivo\npelo microfone,\ngravação nova", COLOR_EXTRA, fontsize=8.2)
    for eb in (e1, e2, e3):
        _arrow(ax, (6, b8[1] - b8[3] / 2), (eb[0], eb[1] + eb[3] / 2), color="#990000", lw=1.1)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig0_pipeline_experimento.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Diagrama salvo em {FIG_DIR / 'fig0_pipeline_experimento.png'}")


if __name__ == "__main__":
    main()
