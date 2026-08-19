"""Gera os gráficos finais para o artigo, a partir das tabelas em results/*.csv.

Um gráfico por achado, escolhido conforme a forma de cada tabela:
  fig1 — achado central: EER por canal, comparando treino sem/com Data
         Augmentation (channel_model_table.csv), um painel por feature.
  fig2 — latência das duas arquiteturas vs. os orçamentos de tempo
         (latency.csv).
  fig3 — tamanho do gap de generalização: locutor conhecido (in-sample,
         speaker_dependent_table.csv) vs. desconhecido (LOSO augmentado,
         channel_model_table.csv), por canal.
  fig4 — voz da pesquisadora antes x depois do retreino, in-sample
         (own_voice_pretrain_table.csv vs. own_voice_ceiling_table.csv),
         com nota sobre a falha na gravação nova (fase Q, não está em
         nenhum CSV — só o resultado impresso no terminal na hora).

Uso:
    python -m src.make_figures
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path("results")
FIG_DIR = RESULTS_DIR / "figures"

CHANNEL_ORDER = ["clean", "gsm", "volte", "vonr"]
CHANNEL_LABELS = {"clean": "Limpo", "gsm": "GSM", "volte": "VoLTE", "vonr": "VoNR (EVS)"}
FEATURE_LABELS = {"mfcc": "MFCC + Random Forest", "wav2vec2": "Wav2Vec2 + Regressão Logística"}


def _channel_x():
    return np.arange(len(CHANNEL_ORDER))


def fig1_eer_canal_regime() -> None:
    df = pd.read_csv(RESULTS_DIR / "channel_model_table.csv")
    x = _channel_x()
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, feature in zip(axes, ["mfcc", "wav2vec2"]):
        sub = df[df["feature"] == feature]
        for i, (regime, label) in enumerate([
            ("clean", "Treinado só com canal limpo"),
            ("augmented", "Treinado com Data Augmentation"),
        ]):
            vals = [sub[(sub["channel"] == c) & (sub["regime"] == regime)]["eer"].iloc[0] for c in CHANNEL_ORDER]
            ax.bar(x + (i - 0.5) * width, vals, width, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels([CHANNEL_LABELS[c] for c in CHANNEL_ORDER])
        ax.set_title(FEATURE_LABELS[feature])
        ax.set_ylim(0, max(df["eer"]) * 1.25)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("EER (quanto menor, melhor)")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Efeito do canal de telecom no EER: treino sem vs. com Data Augmentation")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_eer_canal_regime.png", dpi=150)
    plt.close(fig)


def fig2_latencia() -> None:
    df = pd.read_csv(RESULTS_DIR / "latency.csv")
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.barh(y, df["p95_ms"], color=["#4c72b0", "#dd8452"])
    ax.set_yticks(y)
    ax.set_yticklabels(df["arquitetura"])
    ax.set_xlabel("Latência p95 por janela de 1s (ms)")
    ax.axvline(150, color="crimson", linestyle="--", linewidth=1.5, label="Orçamento de ligação ao vivo (150ms)")
    ax.axvline(1000, color="black", linestyle=":", linewidth=1.5, label="Duração da própria janela (1000ms)")
    for yi, v in zip(y, df["p95_ms"]):
        ax.text(v + 15, yi, f"{v:.0f}ms", va="center")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Latência de inferência (p95) vs. orçamentos de tempo")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_latencia.png", dpi=150)
    plt.close(fig)


def _dumbbell(ax, categories, left_vals, right_vals, left_label, right_label, left_color, right_color, xlabel):
    """Um ponto para cada valor, ligados por uma linha — deixa a DISTÂNCIA entre as
    duas situações (não a altura de uma barra) como o elemento visual principal.
    Evita o problema de uma barra "sumir" quando um dos lados é ~0."""
    y = np.arange(len(categories))
    for yi, l, r in zip(y, left_vals, right_vals):
        ax.plot([l, r], [yi, yi], color="lightgray", zorder=1, linewidth=2.5)
    ax.scatter(left_vals, y, color=left_color, s=140, zorder=2, label=left_label)
    ax.scatter(right_vals, y, color=right_color, s=140, zorder=2, label=right_label)
    for yi, l, r in zip(y, left_vals, right_vals):
        ax.text(l, yi + 0.22, f"{l:.2f}", ha="center", fontsize=8.5, color=left_color, fontweight="bold")
        ax.text(r, yi + 0.22, f"{r:.2f}", ha="center", fontsize=8.5, color=right_color, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(categories)
    ax.set_xlabel(xlabel)
    ax.set_xlim(-0.03, max(max(left_vals), max(right_vals)) * 1.3)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()


def fig3_conhecido_vs_desconhecido() -> None:
    known = pd.read_csv(RESULTS_DIR / "speaker_dependent_table.csv")
    known_by_channel = known.groupby("channel")["eer"].mean()

    unknown = pd.read_csv(RESULTS_DIR / "channel_model_table.csv")
    unknown = unknown[(unknown["feature"] == "mfcc") & (unknown["regime"] == "augmented")].set_index("channel")["eer"]

    labels = [CHANNEL_LABELS[c] for c in CHANNEL_ORDER]
    left_vals = [known_by_channel[c] for c in CHANNEL_ORDER]
    right_vals = [unknown[c] for c in CHANNEL_ORDER]

    fig, ax = plt.subplots(figsize=(9, 5))
    _dumbbell(
        ax, labels, left_vals, right_vals,
        left_label="Locutor conhecido (in-sample: Taylor/Biden/Ryan)",
        right_label="Locutor desconhecido (LOSO, MFCC augmentado)",
        left_color="#2ca02c", right_color="#d62728",
        xlabel="EER (quanto maior, mais errado)",
    )
    ax.set_title("O gap de generalização: erro cresce quando o modelo não conhece a pessoa", fontsize=11.5)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(FIG_DIR / "fig3_conhecido_vs_desconhecido.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig4_voz_propria_antes_depois() -> None:
    before = pd.read_csv(RESULTS_DIR / "own_voice_pretrain_table.csv").set_index("channel")["eer"]
    after = pd.read_csv(RESULTS_DIR / "own_voice_ceiling_table.csv").set_index("channel")["eer"]

    labels = [CHANNEL_LABELS[c] for c in CHANNEL_ORDER]
    before_vals = [before[c] for c in CHANNEL_ORDER]
    after_vals = [after[c] for c in CHANNEL_ORDER]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [2.3, 1]})

    _dumbbell(
        ax_left, labels, after_vals, before_vals,
        left_label="Depois do retreino, MESMOS áudios do treino (in-sample)",
        right_label="Antes do retreino (modelo nunca tinha visto a voz)",
        left_color="#2ca02c", right_color="#d62728",
        xlabel="EER (quanto maior, mais errado)",
    )
    ax_left.set_title("Voz da pesquisadora: antes x depois do retreino (in-sample)", fontsize=11)
    handles, legend_labels = ax_left.get_legend_handles_labels()
    fig.legend(handles, legend_labels, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.35, -0.08), ncol=1, frameon=False)

    ax_right.barh([0], [100], color="#d62728")
    ax_right.set_xlim(0, 100)
    ax_right.set_yticks([0])
    ax_right.set_yticklabels(["Gravação NOVA\n"])
    ax_right.set_xlabel("% das janelas classificadas como FAKE\n(deveriam ser REAL)")
    ax_right.text(50, 0, "15/15 = 100%", ha="center", va="center", color="white", fontweight="bold", fontsize=10)
    ax_right.set_title("Mesmo modelo, dias depois,\nnuma gravação que NÃO entrou no retreino", fontsize=9.5)

    fig.suptitle(
        "O \"0,00\" da esquerda só vale para os áudios que o modelo já viu — numa gravação nova (direita), ele falha de novo",
        fontsize=10, style="italic", y=1.03,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(FIG_DIR / "fig4_voz_propria_antes_depois.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig1_eer_canal_regime()
    fig2_latencia()
    fig3_conhecido_vs_desconhecido()
    fig4_voz_propria_antes_depois()
    print(f"4 gráficos salvos em {FIG_DIR}/")


if __name__ == "__main__":
    main()
