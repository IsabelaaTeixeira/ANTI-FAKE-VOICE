"""Gera os gráficos finais para o artigo, a partir das tabelas em results/*.csv.

Um gráfico por achado:
  fig1  — achado central: EER por canal x regime de treino, com intervalo de
          95%, nos dois datasets de treino (channel_model_table*.csv).
  fig2  — latência das arquiteturas vs. os dois orçamentos de tempo (latency.csv).
  fig3  — locutor conhecido (trecho novo) vs. desconhecido (LOSO), por canal.
  fig4  — voz própria: modelos originais vs. retreinados com a voz.
  fig5  — EER no canal de captura (mic) nos dois datasets.
  fig6  — teste ao vivo com 2 microfones (etapa intermediária do projeto; a repetir).
  fig7  — distribuição de atributos MFCC isolados, real vs. clonada.
  fig8  — EER por camada do Wav2Vec2 (layer_sweep.csv).
  fig9  — decisão por ligação: erro vs. segundos de evidência (call_level_table.csv).
  fig10 — ASVspoof 2021 LA, telefonia real, vs. baselines oficiais.
  fig11 — conjunto em português, por sintetizador.
  fig12 — EER por tipo de ataque no eval do ASVspoof 2019.
  fig13 — por que a acurácia comum engana (classes desbalanceadas).

Cores: cada regime de treino tem sempre a mesma cor (azul = só canal limpo,
laranja = Data Augmentation, verde-água = augmentation pareado), numa paleta
validada para daltonismo. Uma figura que falte dados (etapa ainda não rodada)
é pulada com aviso, sem interromper as outras.

Uso:
    python -m src.make_figures
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features import W2V_LAYER
from src.train import TRAIN_CHANNELS

RESULTS_DIR = Path("results")
FIG_DIR = RESULTS_DIR / "figures"

CHANNEL_ORDER = ["clean", "gsm", "volte", "vonr", "vonr_loss", "mic"]
CHANNEL_LABELS = {
    "clean": "Limpo", "gsm": "GSM", "volte": "VoLTE", "vonr": "VoNR (EVS)",
    "vonr_loss": "VoNR + perda\nde pacotes*", "mic": "Mic/\nambiente",
}
REAL_CHANNEL_ORDER = ["none", "alaw", "ulaw", "g722", "gsm", "opus", "pstn", "todas"]
REAL_CHANNEL_LABELS = {
    "none": "Sem codec", "alaw": "a-law\n(VoIP)", "ulaw": "µ-law\n(VoIP)", "g722": "G.722\n(VoIP)",
    "gsm": "GSM", "opus": "Opus\n(VoIP)", "pstn": "PSTN\n(rede fixa)", "todas": "Todas",
}
FEATURE_LABELS = {"mfcc": "MFCC + Random Forest", "wav2vec2": f"Wav2Vec2 (camada {W2V_LAYER}) + Regressão Logística"}
REGIME_ORDER = ["clean", "augmented", "augmented_matched"]
REGIME_LABELS = {
    "clean": "Treino só com canal limpo",
    "augmented": "Data Augmentation (5 canais)",
    "augmented_matched": "Augmentation pareado (mesmo nº de janelas do limpo)",
}
# paleta categórica validada (scripts/validate_palette.js do guia de visualização): 3 primeiras posições
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
REGIME_COLORS = dict(zip(REGIME_ORDER, SERIES))
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK_SECONDARY, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.size": 9.5, "legend.frameon": False,
})


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / name)


def _save(fig, name: str) -> None:
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvo {FIG_DIR / name}")


def _chance_line(ax, y=0.5, label="acaso") -> None:
    ax.axhline(y, color=INK_SECONDARY, linestyle=":", linewidth=1)
    ax.text(ax.get_xlim()[1], y, f" {label}", va="center", ha="left", fontsize=8, color=INK_SECONDARY)


def _grouped_bars(ax, df: pd.DataFrame, categories: list, cat_col: str, value: str = "eer", ci: bool = True) -> None:
    """Barras agrupadas: uma categoria por grupo, um regime por barra, com intervalo de 95%."""
    regimes = [r for r in REGIME_ORDER if r in set(df["regime"])]
    width = 0.8 / len(regimes)
    x = np.arange(len(categories))
    for i, regime in enumerate(regimes):
        sub = df[df["regime"] == regime].set_index(cat_col).reindex(categories)
        offset = (i - (len(regimes) - 1) / 2) * width
        ax.bar(x + offset, sub[value], width * 0.92, color=REGIME_COLORS[regime], label=REGIME_LABELS[regime])
        if ci and "eer_ci_low" in sub:
            ax.errorbar(
                x + offset, sub[value],
                yerr=[sub[value] - sub["eer_ci_low"], sub["eer_ci_high"] - sub[value]],
                fmt="none", ecolor=INK_SECONDARY, elinewidth=0.9, capsize=2,
            )
    ax.set_xticks(x)


def fig1_eer_canal_regime() -> None:
    dv = _read("channel_model_table.csv")
    asv = _read("channel_model_table_asvspoof.csv")
    asv = asv[asv["split"] == "eval"]
    rows = [("DEEP-VOICE · LOSO (locutor nunca visto)", dv), ("ASVspoof 2019 · eval (locutor e ataque nunca vistos)", asv)]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharey=True)
    for r, (row_title, df) in enumerate(rows):
        for c, feature in enumerate(["mfcc", "wav2vec2"]):
            ax = axes[r, c]
            _grouped_bars(ax, df[df["feature"] == feature], CHANNEL_ORDER, "channel")
            ax.set_xticklabels([CHANNEL_LABELS[ch] for ch in CHANNEL_ORDER], fontsize=8.5)
            ax.set_title(f"{row_title}\n{FEATURE_LABELS[feature]}", fontsize=9.5, loc="left")
            ax.set_ylim(0, 0.62)
            _chance_line(ax)
        axes[r, 0].set_ylabel("EER (menor = melhor)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    fig.text(0.5, -0.06, "Barras de erro: intervalo de 95% (bootstrap por locutor).  *VoNR com perda de pacotes nunca entra no treino.",
             ha="center", fontsize=8.5, color=INK_SECONDARY)
    fig.suptitle("Efeito do canal no erro do detector, por regime de treino", fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, "fig1_eer_canal_regime.png")


def fig2_latencia() -> None:
    df = _read("latency.csv")
    y = np.arange(len(df))[::-1]
    fig, ax = plt.subplots(figsize=(9.5, 0.75 * len(df) + 1.6))
    ax.barh(y, df["p95_ms"], height=0.55, color=SERIES[0])
    ax.set_yticks(y)
    ax.set_yticklabels(df["arquitetura"], fontsize=8.8)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    for yi, v in zip(y, df["p95_ms"]):
        ax.text(v + 12, yi, f"{v:.0f} ms", va="center", fontsize=8.8)
    for x, label, style in [(150, "150 ms: detector NO caminho da voz (ITU-T G.114)", "--"), (1000, "1000 ms: detector em PARALELO à ligação", ":")]:
        ax.axvline(x, color=INK, linestyle=style, linewidth=1.2)
        ax.text(x + 10, y.max() + 0.55, label, fontsize=8, color=INK_SECONDARY, va="bottom")
    ax.set_xlim(0, 1150)
    ax.set_ylim(-0.6, y.max() + 1.0)
    ax.set_xlabel("Latência p95 por janela de 1s, CPU de notebook (ms)")
    ax.set_title("Latência de inferência vs. orçamentos de tempo", loc="left", fontsize=11)
    _save(fig, "fig2_latencia.png")


def _dumbbell(ax, labels, left, right, left_label, right_label, left_ci=None, right_ci=None) -> None:
    y = np.arange(len(labels))
    for yi, l, r in zip(y, left, right):
        ax.plot([l, r], [yi, yi], color=GRID, linewidth=3, zorder=1)
    for vals, ci, color, label in [(left, left_ci, SERIES[0], left_label), (right, right_ci, SERIES[1], right_label)]:
        if ci is not None:
            ax.errorbar(vals, y, xerr=[np.array(vals) - ci[0], ci[1] - np.array(vals)], fmt="none", ecolor=color, elinewidth=1, capsize=2, alpha=0.7)
        ax.scatter(vals, y, color=color, s=70, zorder=3, label=label, edgecolors=SURFACE, linewidths=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)


def fig3_conhecido_vs_desconhecido() -> None:
    known = _read("speaker_dependent_table.csv")
    known = known[known["speaker"] == "todos"]
    unknown = _read("channel_model_table.csv")
    unknown = unknown[unknown["regime"] == "augmented"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for ax, feature in zip(axes, ["mfcc", "wav2vec2"]):
        k = known[known["feature"] == feature].set_index("channel").reindex(CHANNEL_ORDER)
        u = unknown[unknown["feature"] == feature].set_index("channel").reindex(CHANNEL_ORDER)
        _dumbbell(
            ax, [CHANNEL_LABELS[c].replace("\n", " ") for c in CHANNEL_ORDER], k["eer"].values, u["eer"].values,
            "Locutor conhecido, trecho novo", "Locutor nunca visto (LOSO)",
            (k["eer_ci_low"].values, k["eer_ci_high"].values), (u["eer_ci_low"].values, u["eer_ci_high"].values),
        )
        ax.set_title(FEATURE_LABELS[feature], loc="left", fontsize=10)
        ax.set_xlabel("EER (menor = melhor)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("DEEP-VOICE: quanto o erro cresce quando o detector não conhece a pessoa (regime augmented)",
                 fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, "fig3_conhecido_vs_desconhecido.png")


def fig4_voz_propria() -> None:
    before_dv = _read("own_voice_pretrain_table.csv").assign(situacao="Modelo DEEP-VOICE, sem retreino")
    before_asv = _read("own_voice_pretrain_table_asvspoof.csv").assign(situacao="Modelo ASVspoof, sem retreino")
    after = _read("own_voice_ceiling_table.csv").assign(situacao="Retreinado com a voz (teste fora do retreino)")
    df = pd.concat([before_dv, before_asv, after], ignore_index=True)
    situacoes = list(dict.fromkeys(df["situacao"]))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    width = 0.8 / len(situacoes)
    x = np.arange(len(CHANNEL_ORDER))
    for ax, feature in zip(axes, ["mfcc", "wav2vec2"]):
        for i, (situacao, color) in enumerate(zip(situacoes, SERIES)):
            sub = df[(df["feature"] == feature) & (df["situacao"] == situacao)].set_index("channel").reindex(CHANNEL_ORDER)
            ax.bar(x + (i - 1) * width, sub["balanced_accuracy"], width * 0.92, color=color, label=situacao)
        ax.set_xticks(x)
        ax.set_xticklabels([CHANNEL_LABELS[c] for c in CHANNEL_ORDER], fontsize=8.5)
        ax.set_ylim(0, 1.05)
        ax.set_title(FEATURE_LABELS[feature], loc="left", fontsize=10)
        _chance_line(ax)
    axes[0].set_ylabel("Acurácia balanceada\n(média do acerto em real e em clonada)")
    n_test = int(after["n"].iloc[0]) if len(after) else 0
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07))
    fig.text(0.5, -0.11, f"Poucas janelas por canal (retreino: {n_test} janelas de teste por canal) — resultado ilustrativo, não estatístico.",
             ha="center", fontsize=8.5, color=INK_SECONDARY)
    fig.suptitle("Voz da pesquisadora (real + clonada com XTTS v2)", fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, "fig4_voz_propria_antes_depois.png")


def fig5_dataset_comparativo() -> None:
    dv = _read("channel_model_table.csv").assign(grupo="DEEP-VOICE\n(LOSO)")
    asv = _read("channel_model_table_asvspoof.csv")
    asv["grupo"] = asv["split"].map({"dev": "ASVspoof 2019\n(dev, ataques conhecidos)", "eval": "ASVspoof 2019\n(eval, ataques novos)"})
    df = pd.concat([dv, asv], ignore_index=True)
    df = df[(df["feature"] == "wav2vec2") & (df["channel"] == "mic")]
    groups = list(dict.fromkeys(df["grupo"]))

    fig, ax = plt.subplots(figsize=(9, 4.6))
    _grouped_bars(ax, df, groups, "grupo")
    ax.set_xticklabels(groups)
    ax.set_ylabel("EER no canal mic (menor = melhor)")
    ax.set_ylim(0, 0.6)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title(f"Canal de captura (mic/ambiente), {FEATURE_LABELS['wav2vec2']}", loc="left", fontsize=10.5)
    _save(fig, "fig5_dataset_comparativo.png")


def fig6_microfone_generalizacao() -> None:
    """Teste ao vivo: % de janelas de voz real corretas por microfone real.
    Números do teste feito numa etapa intermediária do projeto (sem normalização
    de volume no live_demo, camada 12, codec janela a janela) — não estão em
    nenhum CSV; para atualizar, refaça o teste com src/live_demo.py."""
    mics = ["Microfone do notebook", "Headset close-talk"]
    deepvoice_pct = [80, 0]
    asvspoof_pct = [80, 60]

    y = np.arange(len(mics))
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for yi, l, r in zip(y, deepvoice_pct, asvspoof_pct):
        ax.plot([l, r], [yi, yi], color=GRID, zorder=1, linewidth=3)
    ax.scatter(deepvoice_pct, y, s=340, facecolors="none", edgecolors=SERIES[1], linewidths=3,
               zorder=3, label="Treinado no DEEP-VOICE (6 locutores)")
    ax.scatter(asvspoof_pct, y, color=SERIES[0], s=120, zorder=2, label="Treinado no ASVspoof (107 locutores)")
    for yi, l, r in zip(y, deepvoice_pct, asvspoof_pct):
        ax.text(l, yi + 0.28, f"{l:.0f}%", ha="center", fontsize=9.5, color=INK, fontweight="bold")
        ax.text(r, yi - 0.32, f"{r:.0f}%", ha="center", fontsize=9.5, color=INK, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(mics)
    ax.set_ylim(-0.6, len(mics) - 0.4)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlabel("% das janelas de voz real classificadas corretamente\n(Wav2Vec2, teste ao vivo, 15 janelas por condição)")
    ax.set_xlim(-5, 100)
    ax.axvline(50, color=INK_SECONDARY, linestyle=":", linewidth=1, label="Acaso (50%)")
    ax.set_title("Teste ao vivo com microfones reais — etapa intermediária do projeto (a repetir com o pipeline atual)",
                 fontsize=10, loc="left")
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=1)
    fig.tight_layout(rect=[0, 0.16, 1, 1])
    _save(fig, "fig6_microfone_generalizacao.png")


def fig7_distribuicao_atributos() -> None:
    """Boxplot com jitter mostrando que atributos MFCC individuais NÃO separam
    bem voz real de clonada — a diferença é estatisticamente significativa
    (Mann-Whitney U, p<0,001, por causa do n grande), mas as distribuições se
    sobrepõem muito. É a evidência visual de por que o projeto combina 26
    atributos com um classificador em vez de usar um único limiar."""
    rng = np.random.default_rng(42)
    df = pd.read_csv("data/features_mfcc.csv")
    df_clean = df[df["channel"] == "clean"]

    specs = [("rms", "RMS"), ("spectral_centroid", "Centroide espectral [Hz]")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3))
    for ax, (col, label) in zip(axes, specs):
        real = df_clean[df_clean["label"] == "real"][col].values
        fake = df_clean[df_clean["label"] == "fake"][col].values
        bp = ax.boxplot([real, fake], tick_labels=["Real", "Clonada"], widths=0.5, showfliers=False, patch_artist=True)
        for patch, color in zip(bp["boxes"], ["#cde2fb", "#fbd9cb"]):
            patch.set_facecolor(color)
            patch.set_edgecolor(INK_SECONDARY)
        for element in ("whiskers", "caps", "medians"):
            for line in bp[element]:
                line.set_color(INK_SECONDARY)
        # jitter de uma amostra (não os ~15 mil pontos, senão vira uma mancha ilegível)
        for i, vals in enumerate([real, fake], start=1):
            sample = rng.choice(vals, size=min(35, len(vals)), replace=False)
            ax.scatter(rng.normal(i, 0.06, size=len(sample)), sample, s=10, color=INK_SECONDARY, alpha=0.6, zorder=3)
        ax.set_ylabel(label)
    fig.suptitle("Atributos isolados se sobrepõem entre real e clonada (DEEP-VOICE, canal limpo)", fontsize=11)
    fig.text(0.5, -0.02,
             "Pontos: amostra de 35 observações por classe. Diferença estatisticamente significativa\n"
             "(Mann-Whitney U, p<0,001 nos dois atributos) mas de pequena magnitude — nenhum dos dois separa as classes sozinho.",
             ha="center", fontsize=8, style="italic", color=INK_SECONDARY)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _save(fig, "fig7_distribuicao_atributos.png")


def fig8_camadas_wav2vec2() -> None:
    df = _read("layer_sweep.csv")
    df = df[df["channel"].isin(TRAIN_CHANNELS)].groupby(["dataset", "split", "layer"])["eer"].mean().reset_index()
    series = [
        ("asvspoof", "dev", "ASVspoof 2019 dev (usado para ESCOLHER a camada)"),
        ("asvspoof", "eval", "ASVspoof 2019 eval (confirmação)"),
        ("deepvoice", "loso", "DEEP-VOICE LOSO (confirmação)"),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for (dataset, split, label), color in zip(series, SERIES):
        sub = df[(df["dataset"] == dataset) & (df["split"] == split)].sort_values("layer")
        ax.plot(sub["layer"], sub["eer"], color=color, linewidth=2, marker="o", markersize=5, label=label)
    for layer, text in [(W2V_LAYER, f"escolhida ({W2V_LAYER})"), (12, "última (12),\nusada antes")]:
        ax.axvline(layer, color=INK_SECONDARY, linestyle=":", linewidth=1)
        ax.text(layer + 0.1, ax.get_ylim()[1], text, fontsize=8, color=INK_SECONDARY, va="top")
    ax.set_xticks(range(13))
    ax.set_xlabel("Camada do Wav2Vec2 (0 = saída do bloco convolucional, 12 = última do transformer)")
    ax.set_ylabel("EER médio nos 5 canais de treino")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title("Qual camada do Wav2Vec2 carrega mais pista de voz sintética? (regime augmented)", loc="left", fontsize=10.5)
    _save(fig, "fig8_camadas_wav2vec2.png")


def fig9_decisao_por_ligacao() -> None:
    df = _read("call_level_table.csv")
    df = df[df["regime"] == "augmented"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for feature, color in zip(["mfcc", "wav2vec2"], SERIES):
        for channel, style in [("clean", "-"), ("vonr", "--")]:
            sub = df[(df["feature"] == feature) & (df["channel"] == channel)].sort_values("seconds")
            label = f"{FEATURE_LABELS[feature].split(' + ')[0]} · {CHANNEL_LABELS[channel]}"
            axes[0].plot(sub["seconds"], sub["eer"], style, color=color, linewidth=2, marker="o", markersize=4, label=label)
            axes[0].fill_between(sub["seconds"], sub["eer_ci_low"], sub["eer_ci_high"], color=color, alpha=0.08, linewidth=0)
            axes[1].plot(sub["seconds"], sub["detection_at_5pct_false_alarm"], style, color=color, linewidth=2, marker="o", markersize=4, label=label)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 3, 5, 10, 20, 30])
        ax.set_xticklabels(["1", "2", "3", "5", "10", "20", "30"])
        ax.set_xlabel("Segundos de fala acumulados na decisão")
        ax.grid(axis="x", visible=True)
    axes[0].set_ylabel("EER (menor = melhor)")
    axes[0].set_title("Erro (faixa: intervalo de 95%)", loc="left", fontsize=10)
    axes[1].set_ylabel("Clones detectados")
    axes[1].set_ylim(0, 1.02)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axes[1].set_title("Detecção com 5% de alarme falso", loc="left", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Decisão por ligação: acumular segundos de evidência (DEEP-VOICE, LOSO, regime augmented)",
                 fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, "fig9_decisao_por_ligacao.png")


def fig10_asvspoof2021() -> None:
    df = _read("external_eval_table.csv")
    df = df[(df["test_dataset"] == "asvspoof2021") & (df["train_dataset"] == "asvspoof")
            & (df["feature"] == "wav2vec2") & (df["generator"] == "todos")]
    base = _read("asvspoof2021_baselines.csv")
    fig, ax = plt.subplots(figsize=(13, 5))
    _grouped_bars(ax, df, REAL_CHANNEL_ORDER, "channel", value="eer_utterance", ci=False)
    ax.set_xticklabels([REAL_CHANNEL_LABELS[c] for c in REAL_CHANNEL_ORDER], fontsize=8.8)
    x = np.arange(len(REAL_CHANNEL_ORDER))
    for name, marker in [("LFCC-LCNN", "D"), ("RawNet2", "^")]:
        b = base[base["baseline"] == name].set_index("channel").reindex(REAL_CHANNEL_ORDER)
        # mesmas gravações dos nossos modelos (as com pelo menos 1s de fala; ver external_eval.baselines_2021)
        ax.scatter(x, b["eer_utterance_with_windows"], marker=marker, s=40, color=INK, zorder=4,
                   label=f"Baseline oficial {name} (mesmas gravações)")
    ax.set_ylabel("EER por gravação (menor = melhor)")
    # legenda abaixo do gráfico: dentro dele, cobria a barra do GSM (a mais alta)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8.8, bbox_to_anchor=(0.5, -0.1))
    ax.set_title(f"ASVspoof 2021 LA — telefonia REAL, nunca vista no treino · {FEATURE_LABELS['wav2vec2']}, treinado no ASVspoof 2019",
                 loc="left", fontsize=10.5)
    _save(fig, "fig10_asvspoof2021_telefonia_real.png")


def fig11_portugues() -> None:
    df = _read("external_eval_table.csv")
    df = df[(df["test_dataset"] == "pt") & (df["train_dataset"] == "asvspoof") & (df["feature"] == "wav2vec2")]
    generators = [("xtts_v2", "XTTS v2 (2023)"), ("your_tts", "YourTTS (2022)")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, (generator, title) in zip(axes, generators):
        _grouped_bars(ax, df[df["generator"] == generator], CHANNEL_ORDER, "channel")
        ax.set_xticklabels([CHANNEL_LABELS[c] for c in CHANNEL_ORDER], fontsize=8.5)
        ax.set_ylim(0, 0.62)
        ax.set_title(f"Clones feitos com {title}", loc="left", fontsize=10)
        _chance_line(ax)
    axes[0].set_ylabel("EER (menor = melhor)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle(f"Português (MLS, 20 locutores) — outro idioma e sintetizadores mais novos · {FEATURE_LABELS['wav2vec2']}, treinado no ASVspoof 2019",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, "fig11_portugues.png")


def fig12_ataques() -> None:
    df = _read("attack_table_asvspoof.csv")
    df = df[(df["feature"] == "wav2vec2") & (df["regime"] == "augmented")]
    grid = df.pivot_table(index="attack_id", columns="channel", values="eer").reindex(columns=CHANNEL_ORDER)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    im = ax.imshow(grid.values, cmap="Blues", vmin=0, vmax=0.5, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.3 else INK)
    ax.set_xticks(range(len(CHANNEL_ORDER)))
    ax.set_xticklabels([CHANNEL_LABELS[c] for c in CHANNEL_ORDER], fontsize=8.5)
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels(grid.index)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="EER (mais escuro = pior)")
    ax.set_title(f"EER por tipo de ataque, ASVspoof 2019 eval\n{FEATURE_LABELS['wav2vec2']}, regime augmented", loc="left", fontsize=10.5)
    _save(fig, "fig12_ataques.png")


def fig13_acuracia_enganosa() -> None:
    df = _read("channel_model_table_asvspoof.csv")
    df = df[(df["split"] == "eval") & (df["regime"] == "augmented")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    x = np.arange(len(CHANNEL_ORDER))
    for ax, feature in zip(axes, ["mfcc", "wav2vec2"]):
        sub = df[df["feature"] == feature].set_index("channel").reindex(CHANNEL_ORDER)
        ax.bar(x - 0.2, sub["accuracy"], 0.37, color=SERIES[0], label="Acurácia comum")
        ax.bar(x + 0.2, sub["balanced_accuracy"], 0.37, color=SERIES[1], label="Acurácia balanceada")
        ax.set_xticks(x)
        ax.set_xticklabels([CHANNEL_LABELS[c] for c in CHANNEL_ORDER], fontsize=8.5)
        ax.set_ylim(0, 1.05)
        ax.set_title(FEATURE_LABELS[feature], loc="left", fontsize=10)
        share_fake = sub["n_fake"].iloc[0] / sub["n"].iloc[0]
        ax.axhline(share_fake, color=INK, linestyle="--", linewidth=1)
        ax.text(len(CHANNEL_ORDER) - 0.5, share_fake, f'"fake" para tudo: {share_fake:.0%}', ha="right", va="bottom", fontsize=8)
        _chance_line(ax)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Por que a acurácia comum engana: ~93% das janelas de teste são clonadas (ASVspoof 2019 eval, limiar 0,5)",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, "fig13_acuracia_enganosa.png")


FIGURES = [
    fig1_eer_canal_regime, fig2_latencia, fig3_conhecido_vs_desconhecido, fig4_voz_propria,
    fig5_dataset_comparativo, fig6_microfone_generalizacao, fig7_distribuicao_atributos,
    fig8_camadas_wav2vec2, fig9_decisao_por_ligacao, fig10_asvspoof2021, fig11_portugues,
    fig12_ataques, fig13_acuracia_enganosa,
]


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for fig_fn in FIGURES:
        try:
            fig_fn()
        except FileNotFoundError as err:
            print(f"Pulando {fig_fn.__name__}: falta {err.filename} (etapa ainda não rodada)")


if __name__ == "__main__":
    main()
