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


def fig5_dataset_comparativo() -> None:
    """Compara o ganho de Data Augmentation no canal mic entre os dois datasets:
    DEEP-VOICE (LOSO) vs. ASVspoof (splits dev/eval oficiais)."""
    dv = pd.read_csv(RESULTS_DIR / "channel_model_table.csv")
    av = pd.read_csv(RESULTS_DIR / "channel_model_table_asvspoof.csv")

    def eer_mic(df, regime, split=None):
        sub = df[(df["feature"] == "wav2vec2") & (df["regime"] == regime) & (df["channel"] == "mic")]
        if split is not None:
            sub = sub[sub["split"] == split]
        return sub["eer"].iloc[0]

    groups = ["DEEP-VOICE\n(LOSO)", "ASVspoof\n(dev)", "ASVspoof\n(eval,\nataques novos)"]
    clean_vals = [eer_mic(dv, "clean"), eer_mic(av, "clean", "dev"), eer_mic(av, "clean", "eval")]
    aug_vals = [eer_mic(dv, "augmented"), eer_mic(av, "augmented", "dev"), eer_mic(av, "augmented", "eval")]

    x = np.arange(len(groups))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - width / 2, clean_vals, width, label="Treinado só com canal limpo", color="#d62728")
    ax.bar(x + width / 2, aug_vals, width, label="Treinado com Data Augmentation (incl. canal mic)", color="#2ca02c")
    for xi, v in zip(x - width / 2, clean_vals):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    for xi, v in zip(x + width / 2, aug_vals):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    for xi, cv, av_ in zip(x, clean_vals, aug_vals):
        ax.annotate("", xy=(xi + width / 2, av_ + 0.045), xytext=(xi - width / 2, cv + 0.045),
                    arrowprops=dict(arrowstyle="->", color="gray"))
        ax.text(xi, max(cv, av_) + 0.075, f"{(cv - av_) * 100:+.1f}pp", ha="center", fontsize=8.5, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("EER no canal mic (Wav2Vec2, quanto menor melhor)")
    ax.set_ylim(0, max(clean_vals) * 1.35)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title("O ganho de Data Augmentation no canal de captura é maior no dataset maior")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_dataset_comparativo.png", dpi=150)
    plt.close(fig)


def fig6_microfone_generalizacao() -> None:
    """Teste ao vivo: % de janelas corretas por microfone real, comparando o
    modelo Wav2Vec2 treinado em cada dataset. Dumbbell (não barra) porque um
    dos pontos é 0% — o mesmo problema de visualização que fig3/fig4 já
    tratam. MFCC fica de fora por ser 0% nas 4 condições (citado em texto)."""
    mics = ["Microfone do notebook", "Headset close-talk"]
    deepvoice_pct = [80, 0]
    asvspoof_pct = [80, 60]

    # Não usa _dumbbell aqui porque a linha "notebook" tem os dois valores
    # IGUAIS (80/80) — dois marcadores cheios na mesma posição esconderiam um
    # atrás do outro. Anel + ponto cheio deixa os dois visíveis mesmo quando
    # coincidem.
    y = np.arange(len(mics))
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for yi, l, r in zip(y, deepvoice_pct, asvspoof_pct):
        ax.plot([l, r], [yi, yi], color="lightgray", zorder=1, linewidth=2.5)
    ax.scatter(deepvoice_pct, y, s=340, facecolors="none", edgecolors="#dd8452", linewidths=3,
               zorder=3, label="Treinado no DEEP-VOICE (6 locutores)")
    ax.scatter(asvspoof_pct, y, color="#4c72b0", s=120, zorder=2, label="Treinado no ASVspoof (107 locutores)")
    for yi, l, r in zip(y, deepvoice_pct, asvspoof_pct):
        ax.text(l, yi + 0.28, f"{l:.0f}%", ha="center", fontsize=9.5, color="#dd8452", fontweight="bold")
        ax.text(r, yi - 0.32, f"{r:.0f}%", ha="center", fontsize=9.5, color="#4c72b0", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(mics)
    ax.set_ylim(-0.6, len(mics) - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel("% das janelas de voz real classificadas corretamente\n(Wav2Vec2, teste ao vivo, 15 janelas por condição)")
    ax.set_xlim(-5, 100)
    ax.axvline(50, color="gray", linestyle=":", linewidth=1, label="Acaso (50%)")
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("Generalização para hardware de captura novo: dataset maior generaliza melhor", fontsize=11)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=1, frameon=False)
    fig.tight_layout(rect=[0, 0.16, 1, 1])
    fig.savefig(FIG_DIR / "fig6_microfone_generalizacao.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig7_distribuicao_atributos() -> None:
    """Boxplot com jitter (estilo usado nos artigos de referência) mostrando
    que atributos MFCC individuais NÃO separam bem voz real de clonada — a
    diferença é estatisticamente significativa (Mann-Whitney U, p<0,001,
    por causa do n grande), mas as distribuições se sobrepõem muito. É a
    evidência visual de por que a Seção 3.2 combina 26 atributos com um
    classificador em vez de usar um único limiar."""
    rng = np.random.default_rng(42)
    df = pd.read_csv("data/features_mfcc.csv")
    df_clean = df[df["channel"] == "clean"]

    specs = [("rms", "RMS"), ("spectral_centroid", "Centroide espectral [Hz]")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3))
    for ax, (col, label) in zip(axes, specs):
        real = df_clean[df_clean["label"] == "real"][col].values
        fake = df_clean[df_clean["label"] == "fake"][col].values
        bp = ax.boxplot([real, fake], tick_labels=["Real", "Clonada"], widths=0.5,
                         showfliers=False, patch_artist=True)
        for patch, color in zip(bp["boxes"], ["#c9d9ea", "#e8c9c9"]):
            patch.set_facecolor(color)
            patch.set_edgecolor("#434343")
        for element in ("whiskers", "caps", "medians"):
            for line in bp[element]:
                line.set_color("#434343")

        # jitter de uma amostra (não os ~15 mil pontos, senão vira uma mancha ilegível)
        for i, vals in enumerate([real, fake], start=1):
            sample = rng.choice(vals, size=min(35, len(vals)), replace=False)
            x = rng.normal(i, 0.06, size=len(sample))
            ax.scatter(x, sample, s=10, color="#555", alpha=0.6, zorder=3)

        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Atributos isolados se sobrepõem entre real e clonada (canal limpo)", fontsize=11)
    fig.text(0.5, -0.02,
              "Pontos: amostra de 35 observações por classe. Diferença estatisticamente significativa\n"
              "(Mann-Whitney U, p<0,001 nos dois atributos) mas de pequena magnitude — nenhum dos dois separa as classes sozinho.",
              ha="center", fontsize=8, style="italic", color="#555")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(FIG_DIR / "fig7_distribuicao_atributos.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig1_eer_canal_regime()
    fig2_latencia()
    fig3_conhecido_vs_desconhecido()
    fig4_voz_propria_antes_depois()
    fig5_dataset_comparativo()
    fig6_microfone_generalizacao()
    fig7_distribuicao_atributos()
    print(f"7 gráficos salvos em {FIG_DIR}/")


if __name__ == "__main__":
    main()
