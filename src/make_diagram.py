"""Gera o diagrama do desenho experimental (pipeline completo), para abrir a
seção de Metodologia do artigo. Diferente de make_figures.py, não lê nenhum
resultado — é só a representação visual das etapas do pipeline.

Estilo: painéis com título e borda tracejada agrupando cada grande etapa,
conectados por setas-chevron largas — inspirado no estilo de diagrama de
pipeline usado em vários dos artigos de referência (etapas numeradas dentro
de painéis, não uma sequência solta de caixas).

Uso:
    python -m src.make_diagram
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

FIG_DIR = Path("results/figures")

PANEL_BG = "#f3f5f9"
PANEL_EDGE = "#8a97ad"
PANEL_TITLE = "#1f3864"

COLOR_MAIN = "#dbe9fb"
COLOR_CHANNEL = "#e3f1db"
COLOR_FEATURE = "#fdf1cf"
COLOR_RESULT = "#c9e6c2"
COLOR_EXTRA = "#fbdede"
BOX_EDGE = "#54606e"
CHEVRON_COLOR = "#5b7fa6"

plt.rcParams["font.family"] = "sans-serif"


def _box(ax, x, y, w, h, text, color, fontsize=9.5, edge=BOX_EDGE):
    # sombra sutil
    ax.add_patch(FancyBboxPatch(
        (x - w / 2 + 0.06, y - h / 2 - 0.06), w, h,
        boxstyle="round,pad=0.07,rounding_size=0.16",
        linewidth=0, facecolor="#00000018", zorder=1,
    ))
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.07,rounding_size=0.16",
        linewidth=1.3, edgecolor=edge, facecolor=color, zorder=2,
    ))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, zorder=3, linespacing=1.4, color="#20242b")
    return (x, y, w, h)


def _arrow(ax, start, end, **kwargs):
    style = dict(arrowstyle="-|>", lw=1.4, color="#5a6472", shrinkA=0, shrinkB=0)
    style.update(kwargs)
    ax.annotate("", xy=end, xytext=start, arrowprops=style, zorder=1)


def _panel(ax, x0, x1, y0, y1, title):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0.12,rounding_size=0.22",
        linewidth=1.5, linestyle=(0, (5, 3)), edgecolor=PANEL_EDGE, facecolor=PANEL_BG, zorder=0,
    ))
    ax.text(x0 + 0.35, y1 - 0.42, title, ha="left", va="center", fontsize=12.5,
            fontweight="bold", color=PANEL_TITLE, zorder=3)


def _chevron(ax, cx, y_top, y_bottom, w=1.0):
    """Seta larga (chevron) apontando pra baixo, conectando dois painéis."""
    stem_w = w * 0.45
    head_w = w
    head_h = (y_top - y_bottom) * 0.45
    pts = [
        (cx - stem_w / 2, y_top), (cx + stem_w / 2, y_top),
        (cx + stem_w / 2, y_bottom + head_h), (cx + head_w / 2, y_bottom + head_h),
        (cx, y_bottom), (cx - head_w / 2, y_bottom + head_h),
        (cx - stem_w / 2, y_bottom + head_h),
    ]
    ax.add_patch(Polygon(pts, closed=True, facecolor=CHEVRON_COLOR, edgecolor="none", zorder=1, alpha=0.85))


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.5, 17))
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-1.5, 21.5)
    ax.axis("off")

    # ===== Painel 1: Dados e canais =====
    _panel(ax, -0.2, 12.2, 15.6, 21.1, "1 · Dados e canais simulados")
    b1 = _box(ax, 6, 20.0, 9.2, 1.15, "DEEP-VOICE (6 locutores)   ou   ASVspoof 2019 LA (107 locutores)", COLOR_MAIN, fontsize=9)
    b3 = _box(ax, 6, 18.15, 8.7, 1.35, "Fatiamento em janelas de 1s, 16kHz mono\n+ corte de silêncio de borda e normalização", COLOR_MAIN, fontsize=8.6)
    _arrow(ax, (b1[0], b1[1] - b1[3] / 2), (b3[0], b3[1] + b3[3] / 2))

    channel_y = 16.5
    channel_defs = [
        ("Limpo", 1.5), ("GSM\n(AMR-NB 12,2kb/s)", 3.75), ("VoLTE\n(AMR-WB 12,65kb/s)", 6.0), ("VoNR\n(EVS 13,2kb/s)", 8.25), ("Mic/ambiente\n(2 perfis)", 10.5),
    ]
    channel_boxes = [_box(ax, x, channel_y, 2.05, 1.35, label, COLOR_CHANNEL, fontsize=7.6) for label, x in channel_defs]
    for cb in channel_boxes:
        _arrow(ax, (6, b3[1] - b3[3] / 2), (cb[0], cb[1] + cb[3] / 2))
    ax.text(1.5, channel_y + 1.0, "canal de transmissão", ha="center", fontsize=7.2, style="italic", color="#5a6472")
    ax.text(10.5, channel_y + 1.0, "canal de\ncaptura", ha="center", fontsize=7.2, style="italic", color="#5a6472")

    # ===== chevron 1 -> 2 =====
    _chevron(ax, 6, 15.55, 14.75, w=1.4)

    # ===== Painel 2: Atributos e treino =====
    _panel(ax, -0.2, 12.2, 6.3, 14.65, "2 · Atributos, split e regimes de treino")

    bus_y = 13.55
    ax.plot([1.5, 10.5], [bus_y, bus_y], color="#5a6472", lw=1.3, zorder=1)
    for cb in channel_boxes:
        _arrow(ax, (cb[0], cb[1] - cb[3] / 2), (cb[0], bus_y), color="#5a6472")

    feature_y = 12.3
    f1 = _box(ax, 4, feature_y, 5.0, 1.3, "MFCC\n26 atributos espectrais\n(Random Forest)", COLOR_FEATURE, fontsize=8.6)
    f2 = _box(ax, 8.6, feature_y, 5.0, 1.3, "Wav2Vec2\nembedding 768-dim pré-treinado\n(Regressão Logística)", COLOR_FEATURE, fontsize=8.6)
    _arrow(ax, (4, bus_y), (f1[0], f1[1] + f1[3] / 2))
    _arrow(ax, (8.6, bus_y), (f2[0], f2[1] + f2[3] / 2))

    loso_y = 10.5
    b6 = _box(ax, 6, loso_y, 8.3, 1.35, "Split por locutor: Leave-One-Speaker-Out (DEEP-VOICE)\nou split oficial treino/dev/eval (ASVspoof)", COLOR_MAIN, fontsize=8.3)
    _arrow(ax, (4, f1[1] - f1[3] / 2), (5, b6[1] + b6[3] / 2))
    _arrow(ax, (8.6, f2[1] - f2[3] / 2), (7, b6[1] + b6[3] / 2))

    train_y = 8.15
    t1 = _box(ax, 3.3, train_y, 5.3, 1.35, "Regime \"limpo\"\ntreina só com canal limpo", COLOR_MAIN, fontsize=8.6)
    t2 = _box(ax, 8.7, train_y, 5.3, 1.35, "Regime \"augmentado\"\ntreina com todos os canais juntos\n(3 de codec + 1 de captura)", COLOR_MAIN, fontsize=8.0)
    _arrow(ax, (4.5, b6[1] - b6[3] / 2), (t1[0], t1[1] + t1[3] / 2))
    _arrow(ax, (7.5, b6[1] - b6[3] / 2), (t2[0], t2[1] + t2[3] / 2))

    # ===== chevron 2 -> 3 =====
    _chevron(ax, 6, 6.75, 5.85, w=1.4)

    # ===== Painel 3: Avaliação =====
    _panel(ax, -0.2, 12.2, 4.05, 5.7, "3 · Avaliação")
    eval_y = 4.85
    b8 = _box(ax, 6, eval_y, 8.3, 1.2, "Acurácia, F1 e EER por canal,\ntestando sempre no locutor de fora", COLOR_RESULT, fontsize=8.8)
    _arrow(ax, (t1[0], t1[1] - t1[3] / 2), (4.8, b8[1] + b8[3] / 2))
    _arrow(ax, (t2[0], t2[1] - t2[3] / 2), (7.2, b8[1] + b8[3] / 2))

    # ===== chevron 3 -> 4 =====
    _chevron(ax, 6, 3.75, 3.0, w=1.4)

    # ===== Painel 4: Extensões exploratórias =====
    _panel(ax, -0.2, 12.2, -1.2, 3.1, "4 · Extensões exploratórias (adicionadas durante o projeto)")
    ext_y = 0.85
    ext_defs = [
        ("Locutor conhecido\n(Taylor/Biden/Ryan\nin-sample — \"teto\")", 1.3),
        ("Voz própria + XTTS v2\nantes x depois do\nretreino, in-sample", 4.5),
        ("Teste ao vivo,\n2 microfones\n(notebook, headset)", 7.7),
        ("Validação cruzada:\nmesmo teste nos\n2 datasets", 10.9),
    ]
    ext_boxes = [_box(ax, x, ext_y, 2.85, 1.6, label, COLOR_EXTRA, fontsize=7.5) for label, x in ext_defs]
    for eb in ext_boxes:
        _arrow(ax, (6, b8[1] - b8[3] / 2 - 0.15), (eb[0], eb[1] + eb[3] / 2), color="#a34a4a", lw=1.1)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig0_pipeline_experimento.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Diagrama salvo em {FIG_DIR / 'fig0_pipeline_experimento.png'}")


if __name__ == "__main__":
    main()
