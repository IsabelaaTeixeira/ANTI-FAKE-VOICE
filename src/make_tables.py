"""Gera as tabelas do artigo como imagens (não markdown), no estilo
acadêmico "booktabs" (só regras horizontais, sem cor, sem linhas verticais)
usado nos artigos de referência deste trabalho. Os números vêm de
results/*.csv (conferidos manualmente ao montar cada tabela) — este script
não recalcula nada, só formata.

Uso:
    python -m src.make_tables
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = Path("results/figures")
plt.rcParams["font.family"] = "serif"

RULE_COLOR = "black"
ROW_H_IN = 0.28   # altura de cada linha, em polegadas
TITLE_H_IN = 0.34
HEADER_H_IN = 0.32
FONT = 9.5


def _table_image(
    number: int, caption: str, col_labels: list[str], rows: list[list[str]], out_name: str,
    col_widths: list[float] | None = None, width_in: float = 9.0,
) -> None:
    n_cols = len(col_labels)
    n_rows = len(rows)
    total_h = TITLE_H_IN + HEADER_H_IN + ROW_H_IN * n_rows + 0.12
    fig, ax = plt.subplots(figsize=(width_in, total_h))
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, 1)
    # coordenadas em polegadas de topo para baixo, convertidas depois pra fração de eixo
    ax.set_ylim(0, total_h)

    if col_widths is None:
        col_widths = [1.0 / n_cols] * n_cols
    lefts = [sum(col_widths[:i]) for i in range(n_cols)]
    centers = [lefts[i] + col_widths[i] / 2 for i in range(n_cols)]

    y_title = total_h - TITLE_H_IN / 2
    # título: "Tabela N." em negrito seguido da legenda em texto normal, na mesma linha
    t1 = ax.text(0, y_title, f"Tabela {number}.", fontsize=FONT + 1, fontweight="bold", va="center", ha="left")
    fig.canvas.draw()
    bbox = t1.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(ax.transData.inverted())
    ax.text(bbox.x1 + 0.012, y_title, caption, fontsize=FONT + 1, va="center", ha="left")

    body_top = total_h - TITLE_H_IN
    y_header = body_top - HEADER_H_IN / 2
    for label, cx in zip(col_labels, centers):
        ax.text(cx, y_header, label, fontsize=FONT, fontweight="bold", va="center", ha="center")

    y_rule_top = body_top
    y_rule_mid = body_top - HEADER_H_IN
    y_rule_bottom = y_rule_mid - ROW_H_IN * n_rows

    for r, row in enumerate(rows):
        y = y_rule_mid - ROW_H_IN * (r + 0.5)
        for c, (val, cx, left) in enumerate(zip(row, centers, lefts)):
            if c == 0:
                ax.text(left, y, val, fontsize=FONT, va="center", ha="left")
            else:
                ax.text(cx, y, val, fontsize=FONT, va="center", ha="center")

    ax.plot([0, 1], [y_rule_top, y_rule_top], color=RULE_COLOR, lw=1.6, solid_capstyle="butt")
    ax.plot([0, 1], [y_rule_mid, y_rule_mid], color=RULE_COLOR, lw=0.8, solid_capstyle="butt")
    ax.plot([0, 1], [y_rule_bottom, y_rule_bottom], color=RULE_COLOR, lw=1.6, solid_capstyle="butt")

    fig.savefig(FIG_DIR / out_name, dpi=150, pad_inches=0.08)
    plt.close(fig)
    print(f"Salvo {FIG_DIR / out_name}")


def tabela1_dataset_comparativo() -> None:
    _table_image(
        1, "DEEP-VOICE vs. ASVspoof 2019 LA, escala e diversidade.",
        ["", "DEEP-VOICE", "ASVspoof 2019 LA (amostrado)"],
        [
            ["Locutores", "6", "107"],
            ["Tipos de ataque/conversão", "1 (RVC)", "19 (A01–A19: TTS e VC diversos)"],
            ["Split", "LOSO manual (6 dobras)", "Oficial: treino/dev (A01–A06) / eval (A07–A19, nunca vistos)"],
            ["Janelas limpas (após corte de silêncio)", "15.234", "35.374"],
            ["Janelas totais (5 canais)", "76.170", "176.870"],
            ["Corpus original (antes da amostragem)", "~15,3 mil gravações (uso integral)", "~121,5 mil gravações (amostrado, teto 20/grupo)"],
        ],
        "tabela1_dataset_comparativo.png",
        col_widths=[0.28, 0.24, 0.48],
        width_in=11,
    )


def tabela2_eer_mic_deepvoice() -> None:
    _table_image(
        2, "EER no canal mic, DEEP-VOICE.",
        ["Atributo", "Regime", "EER", "Δ com Data Augmentation"],
        [
            ["MFCC", "limpo", "0,380", "—"],
            ["MFCC", "augmentado", "0,388", "+0,8pp (piora)"],
            ["Wav2Vec2", "limpo", "0,314", "—"],
            ["Wav2Vec2", "augmentado", "0,302", "-1,2pp (melhora)"],
        ],
        "tabela2_eer_mic_deepvoice.png",
        col_widths=[0.22, 0.22, 0.16, 0.32],
        width_in=8.5,
    )


def tabela3_eer_asvspoof_canais() -> None:
    _table_image(
        3, "EER por canal, Wav2Vec2, ASVspoof (regime augmentado).",
        ["Split", "Limpo", "GSM", "VoLTE", "VoNR", "Mic"],
        [
            ["dev (ataques conhecidos)", "0,106", "0,156", "0,121", "0,133", "0,214"],
            ["eval (ataques novos)", "0,210", "0,254", "0,233", "0,239", "0,311"],
        ],
        "tabela3_eer_asvspoof_canais.png",
        col_widths=[0.28, 0.14, 0.14, 0.14, 0.14, 0.14],
        width_in=9.5,
    )


def tabela4_teste_ao_vivo() -> None:
    _table_image(
        4, "% de janelas de voz real classificadas corretamente, teste ao vivo (15 janelas/condição).",
        ["Microfone", "Wav2Vec2 / DEEP-VOICE", "Wav2Vec2 / ASVspoof", "MFCC (ambos os datasets)"],
        [
            ["Notebook (embutido)", "80%", "80%", "0%"],
            ["Headset (close-talk)", "0%", "60%", "0%"],
        ],
        "tabela4_teste_ao_vivo.png",
        width_in=10,
    )


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tabela1_dataset_comparativo()
    tabela2_eer_mic_deepvoice()
    tabela3_eer_asvspoof_canais()
    tabela4_teste_ao_vivo()
    print(f"4 tabelas salvas em {FIG_DIR}/")


if __name__ == "__main__":
    main()
