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

from src.features import W2V_LAYER

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
    fig, ax = plt.subplots(figsize=(11.5, 19))
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-3.6, 21.5)
    ax.axis("off")

    # ===== Painel 1: Dados e canais =====
    _panel(ax, -0.2, 12.2, 15.6, 21.1, "1 · Dados e canais simulados")
    b1 = _box(ax, 6, 19.75, 9.2, 1.0, "DEEP-VOICE (6 locutores)   ou   ASVspoof 2019 LA (107 locutores)", COLOR_MAIN, fontsize=9)
    b3 = _box(ax, 6, 18.1, 9.6, 1.25, "Corte do silêncio de borda → canal aplicado na GRAVAÇÃO INTEIRA (codec contínuo)\n→ janelas de 1s, 16kHz mono, cada uma normalizada a -16 LUFS", COLOR_MAIN, fontsize=8.3)
    _arrow(ax, (b1[0], b1[1] - b1[3] / 2), (b3[0], b3[1] + b3[3] / 2))

    channel_y = 16.5
    channel_defs = [
        ("Limpo", 1.0), ("GSM\n(AMR-NB\n12,2kb/s)", 3.0), ("VoLTE\n(AMR-WB\n12,65kb/s)", 5.0),
        ("VoNR\n(EVS\n13,2kb/s)", 7.0), ("VoNR + perda\nde pacotes\n(só teste)", 9.0), ("Mic/ambiente\n(2 perfis)", 11.0),
    ]
    channel_boxes = [_box(ax, x, channel_y, 1.75, 1.35, label, COLOR_CHANNEL, fontsize=7.2) for label, x in channel_defs]
    for cb in channel_boxes:
        _arrow(ax, (6, b3[1] - b3[3] / 2), (cb[0], cb[1] + cb[3] / 2))

    # ===== chevron 1 -> 2 =====
    _chevron(ax, 6, 15.55, 14.75, w=1.4)

    # ===== Painel 2: Atributos e treino =====
    _panel(ax, -0.2, 12.2, 6.95, 14.65, "2 · Atributos, split e regimes de treino")

    bus_y = 13.55
    ax.plot([1.0, 11.0], [bus_y, bus_y], color="#5a6472", lw=1.3, zorder=1)
    for cb in channel_boxes:
        _arrow(ax, (cb[0], cb[1] - cb[3] / 2), (cb[0], bus_y), color="#5a6472")

    feature_y = 12.3
    f1 = _box(ax, 3.4, feature_y, 5.0, 1.3, "MFCC\n26 atributos espectrais\n(Random Forest)", COLOR_FEATURE, fontsize=8.6)
    f2 = _box(ax, 8.8, feature_y, 5.4, 1.3, f"Wav2Vec2, camada {W2V_LAYER} (de 0 a 12)\n(escolhida só no dev do ASVspoof)\n(Regressão Logística)", COLOR_FEATURE, fontsize=8.4)
    _arrow(ax, (3.4, bus_y), (f1[0], f1[1] + f1[3] / 2))
    _arrow(ax, (8.8, bus_y), (f2[0], f2[1] + f2[3] / 2))

    loso_y = 10.5
    b6 = _box(ax, 6, loso_y, 8.3, 1.35, "Split por locutor: Leave-One-Speaker-Out (DEEP-VOICE)\nou split oficial treino/dev/eval (ASVspoof)", COLOR_MAIN, fontsize=8.3)
    _arrow(ax, (3.4, f1[1] - f1[3] / 2), (5, b6[1] + b6[3] / 2))
    _arrow(ax, (8.8, f2[1] - f2[3] / 2), (7, b6[1] + b6[3] / 2))

    train_y = 8.15
    regimes = [
        ("Regime \"limpo\"\nsó o canal limpo", 2.1),
        ("Regime \"augmentado\"\nos 5 canais de treino\n(5x mais janelas)", 6.0),
        ("Regime \"augmentado pareado\"\nos 5 canais, mesmo nº\nde janelas do limpo", 9.9),
    ]
    train_boxes = [_box(ax, x, train_y, 3.6, 1.45, label, COLOR_MAIN, fontsize=7.9) for label, x in regimes]
    for tb in train_boxes:
        _arrow(ax, (6, b6[1] - b6[3] / 2), (tb[0], tb[1] + tb[3] / 2))

    # ===== chevron 2 -> 3 =====
    _chevron(ax, 6, 6.95, 6.05, w=1.4)

    # ===== Painel 3: Avaliação =====
    _panel(ax, -0.2, 12.2, 3.85, 6.0, "3 · Avaliação")
    eval_y = 4.55
    _box(ax, 6, eval_y, 10.2, 1.25, "EER com intervalo de 95% (bootstrap por locutor), acurácia balanceada e min t-DCF (ASVspoof),\npor canal, por tipo de ataque e por locutor — sempre em locutores fora do treino", COLOR_RESULT, fontsize=8.2)

    # ===== chevron 3 -> 4 =====
    _chevron(ax, 6, 3.6, 2.85, w=1.4)

    # ===== Painel 4: Testes complementares =====
    _panel(ax, -0.2, 12.2, -3.4, 2.95, "4 · Testes complementares")
    ext_defs = [
        ("Locutor conhecido,\ntrecho novo\n(70% treino / 25% teste)", 1.95, 0.85),
        ("Decisão por ligação\n(evidência acumulada\nem 1 a 30 s)", 6.0, 0.85),
        ("ASVspoof 2021 LA:\ntelefonia REAL\n(7 condições)", 10.05, 0.85),
        ("Português (MLS)\nclonado por\nXTTS v2 e YourTTS", 1.95, -1.85),
        ("Voz própria +\nteste ao vivo\npelo microfone", 6.0, -1.85),
        ("Latência: modelo\ncortado na camada\ne quantizado (int8)", 10.05, -1.85),
    ]
    for label, x, y in ext_defs:
        _box(ax, x, y, 3.5, 1.5, label, COLOR_EXTRA, fontsize=8.2)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig0_pipeline_experimento.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Diagrama salvo em {FIG_DIR / 'fig0_pipeline_experimento.png'}")


if __name__ == "__main__":
    main()
