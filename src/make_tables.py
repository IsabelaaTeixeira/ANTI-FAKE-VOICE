"""Gera as tabelas do artigo como imagens (não markdown), no estilo
acadêmico "booktabs" (só regras horizontais, sem cor, sem linhas verticais)
usado nos artigos de referência deste trabalho. Os números são lidos de
results/*.csv (e as contagens de janelas, dos manifests em data/) — nada é
digitado à mão, exceto a tabela do teste ao vivo, que não tem CSV. Uma
tabela cujos dados ainda não existem é pulada com aviso.

Uso:
    python -m src.make_tables
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.features import W2V_LAYER

FIG_DIR = Path("results/figures")
RESULTS_DIR = Path("results")
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


CHANNEL_ORDER = ["clean", "gsm", "volte", "vonr", "vonr_loss", "mic"]
CHANNEL_LABELS = {
    "clean": "Limpo", "gsm": "GSM", "volte": "VoLTE", "vonr": "VoNR (EVS)",
    "vonr_loss": "VoNR + perda de pacotes*", "mic": "Mic/ambiente",
}
REAL_CHANNEL_ORDER = ["none", "alaw", "ulaw", "g722", "gsm", "opus", "pstn", "todas"]
REAL_CHANNEL_LABELS = {
    "none": "Sem codec", "alaw": "a-law (VoIP)", "ulaw": "µ-law (VoIP)", "g722": "G.722 (VoIP)",
    "gsm": "GSM", "opus": "Opus (VoIP)", "pstn": "PSTN (rede fixa)", "todas": "Todas juntas",
}
REGIME_SHORT = {"clean": "limpo", "augmented": "augmentado", "augmented_matched": "pareado"}
W2V = f"Wav2Vec2 (camada {W2V_LAYER})"


def _num(v: float, digits: int = 3) -> str:
    return "—" if pd.isna(v) else f"{v:.{digits}f}".replace(".", ",")


def _pct(v: float) -> str:
    return "—" if pd.isna(v) else f"{v:.0%}"


def _eer_ci(row: pd.Series) -> str:
    return f"{_num(row['eer'])} [{_num(row['eer_ci_low'], 2)}–{_num(row['eer_ci_high'], 2)}]"


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / name)


def _count_windows(manifest: str, **filters) -> pd.DataFrame:
    df = pd.read_csv(manifest, usecols=lambda c: c in {"channel", "label", "speakers", "source_file", "split", "attack_id", "generator"})
    for col, val in filters.items():
        df = df[df[col] == val]
    return df


def tabela1_datasets() -> None:
    dv = _count_windows("data/manifest.csv", channel="clean")
    asv = _count_windows("data/manifest_asvspoof.csv", channel="clean")
    a21 = pd.read_csv("data/features_wav2vec2_meta_asvspoof2021.csv", usecols=["speakers", "source_file", "channel", "label"])
    # gravações do eval de 2021 no total (as com menos de 1s de fala não geram janela)
    a21_total = (pd.read_csv("data/raw/ASVSPOOF2021/keys/LA/CM/trial_metadata.txt", sep=" ", header=None, usecols=[7])[7] == "eval").sum()
    pt = _count_windows("data/manifest_pt.csv", channel="clean")

    def windows(df):
        return f"{len(df):,} ({(df['label'] == 'real').sum():,} reais)".replace(",", ".")

    _table_image(
        1, "Conjuntos de dados usados (janelas de 1s; nos simulados, por canal).",
        ["", "DEEP-VOICE", "ASVspoof 2019 LA", "ASVspoof 2021 LA", "Português (MLS)"],
        [
            ["Papel", "treino e teste (LOSO)", "treino / dev / eval", "só teste", "só teste"],
            ["Locutores", str(dv["speakers"].str.split(",").explode().nunique()), str(asv["speakers"].nunique()),
             str(a21["speakers"].nunique()), str(pt["speakers"].nunique())],
            ["Gravações", f"{dv['source_file'].nunique():,}".replace(",", "."), f"{asv['source_file'].nunique():,}".replace(",", "."),
             f"{a21['source_file'].nunique():,} de {a21_total:,}".replace(",", "."), f"{pt['source_file'].nunique():,}".replace(",", ".")],
            ["Ataques", "1 (conversão RVC)", "19 (A01–A19)", "13 (A07–A19)", "2 (XTTS v2, YourTTS)"],
            ["Canais", "6 simulados", "6 simulados", "7 reais (transmitidos)", "6 simulados"],
            ["Janelas", windows(dv), windows(asv), windows(a21), windows(pt)],
        ],
        "tabela1_dataset_comparativo.png",
        col_widths=[0.13, 0.2, 0.21, 0.23, 0.23], width_in=13,
    )


def tabela2_eer_principal() -> None:
    dv = _read("channel_model_table.csv")
    asv = _read("channel_model_table_asvspoof.csv")
    asv = asv[asv["split"] == "eval"]
    rows = []
    for ch in CHANNEL_ORDER:
        row = [CHANNEL_LABELS[ch]]
        for df in (dv, asv):
            for regime in REGIME_SHORT:
                match = df[(df["feature"] == "wav2vec2") & (df["regime"] == regime) & (df["channel"] == ch)]
                row.append(_eer_ci(match.iloc[0]) if len(match) else "—")
        rows.append(row)
    _table_image(
        2, f"EER [intervalo de 95%] por canal e regime de treino, {W2V}. *Canal nunca usado no treino.",
        ["Canal"] + [f"DEEP-VOICE · {r}" for r in REGIME_SHORT.values()] + [f"ASVspoof eval · {r}" for r in REGIME_SHORT.values()],
        rows, "tabela2_eer_principal.png",
        col_widths=[0.16] + [0.14] * 6, width_in=15,
    )


def tabela3_asvspoof_detalhe() -> None:
    df = _read("channel_model_table_asvspoof.csv")
    df = df[(df["split"] == "eval") & (df["feature"] == "wav2vec2")]
    rows = []
    for ch in CHANNEL_ORDER:
        row = [CHANNEL_LABELS[ch]]
        for regime in ("clean", "augmented"):
            r = df[(df["regime"] == regime) & (df["channel"] == ch)].iloc[0]
            row += [_num(r["eer"]), _num(r["min_tdcf"]), _pct(r["balanced_accuracy_dev_threshold"])]
        rows.append(row)
    _table_image(
        3, f"ASVspoof 2019 eval, {W2V}: EER por janela, min t-DCF (por gravação) e acurácia balanceada no limiar do dev.",
        ["Canal", "EER · limpo", "t-DCF · limpo", "Ac. bal. · limpo", "EER · augm.", "t-DCF · augm.", "Ac. bal. · augm."],
        rows, "tabela3_eer_asvspoof_canais.png",
        col_widths=[0.22] + [0.13] * 6, width_in=12.5,
    )


def tabela4_teste_ao_vivo() -> None:
    _table_image(
        4, "% de janelas de voz real classificadas corretamente, teste ao vivo (15 janelas/condição) — etapa intermediária do projeto.",
        ["Microfone", "Wav2Vec2 / DEEP-VOICE", "Wav2Vec2 / ASVspoof", "MFCC (ambos os datasets)"],
        [
            ["Notebook (embutido)", "80%", "80%", "0%"],
            ["Headset (close-talk)", "0%", "60%", "0%"],
        ],
        "tabela4_teste_ao_vivo.png",
        width_in=11,
    )


def tabela5_asvspoof2021() -> None:
    df = _read("external_eval_table.csv")
    df = df[(df["test_dataset"] == "asvspoof2021") & (df["train_dataset"] == "asvspoof") & (df["generator"] == "todos")]
    base = _read("asvspoof2021_baselines.csv")
    rows = []
    for ch in REAL_CHANNEL_ORDER:
        row = [REAL_CHANNEL_LABELS[ch]]
        for feature in ("mfcc", "wav2vec2"):
            for regime in ("clean", "augmented"):
                r = df[(df["feature"] == feature) & (df["regime"] == regime) & (df["channel"] == ch)]
                row.append(_num(r["eer_utterance"].iloc[0]) if len(r) else "—")
        for name in ("LFCC-LCNN", "RawNet2"):
            b = base[(base["baseline"] == name) & (base["channel"] == ch)]
            row.append(_num(b["eer_utterance_with_windows"].iloc[0]) if len(b) else "—")
        rows.append(row)
    _table_image(
        5, f"ASVspoof 2021 LA (telefonia real): EER por gravação. Treino no ASVspoof 2019, sem retreino; "
           f"Wav2Vec2 na camada {W2V_LAYER}; baselines oficiais nas mesmas gravações.",
        ["Condição", "MFCC · limpo", "MFCC · augm.", "Wav2Vec2 · limpo", "Wav2Vec2 · augm.", "LFCC-LCNN (oficial)", "RawNet2 (oficial)"],
        rows, "tabela5_asvspoof2021.png",
        col_widths=[0.16, 0.12, 0.12, 0.16, 0.16, 0.14, 0.14], width_in=14,
    )


def tabela6_portugues() -> None:
    df = _read("external_eval_table.csv")
    df = df[(df["test_dataset"] == "pt") & (df["train_dataset"] == "asvspoof") & (df["feature"] == "wav2vec2")]
    rows = []
    for ch in CHANNEL_ORDER:
        row = [CHANNEL_LABELS[ch]]
        for generator in ("xtts_v2", "your_tts"):
            for regime in ("clean", "augmented"):
                r = df[(df["generator"] == generator) & (df["regime"] == regime) & (df["channel"] == ch)]
                row.append(_eer_ci(r.iloc[0]) if len(r) else "—")
        rows.append(row)
    _table_image(
        6, f"Português (MLS): EER [95%] por sintetizador, {W2V} treinado no ASVspoof 2019, sem retreino.",
        ["Canal", "XTTS v2 · limpo", "XTTS v2 · augm.", "YourTTS · limpo", "YourTTS · augm."],
        rows, "tabela6_portugues.png",
        col_widths=[0.2, 0.2, 0.2, 0.2, 0.2], width_in=12,
    )


def tabela7_latencia() -> None:
    lat = _read("latency.csv")
    quality = _read("latency_quality.csv").set_index("precisao")["eer_dev_clean"]
    rows = [
        [r["arquitetura"], _num(r["mediana_ms"], 0), _num(r["p95_ms"], 0), _num(r["fator_tempo_real_p95"], 2),
         "sim" if r["cabe_em_paralelo_1s"] else "não", "sim" if r["cabe_no_caminho_da_voz_150ms"] else "não"]
        for _, r in lat.iterrows()
    ]
    _table_image(
        7, f"Latência por janela de 1s (CPU de notebook). EER no dev do ASVspoof, canal limpo: float32 {_num(quality.get('float32'))}, int8 {_num(quality.get('int8'))}.",
        ["Arquitetura", "Mediana (ms)", "p95 (ms)", "Fator tempo real", "Em paralelo (<1s)", "No caminho da voz (<150ms)"],
        rows, "tabela7_latencia.png",
        col_widths=[0.34, 0.1, 0.09, 0.12, 0.15, 0.2], width_in=14,
    )


def tabela8_decisao_por_ligacao() -> None:
    df = _read("call_level_table.csv")
    df = df[df["regime"] == "augmented"]
    seconds = [1, 3, 10, 30]
    rows = []
    for feature, label in (("mfcc", "MFCC"), ("wav2vec2", W2V)):
        for ch in ("clean", "vonr", "vonr_loss"):
            sub = df[(df["feature"] == feature) & (df["channel"] == ch)].set_index("seconds")
            row = [f"{label} · {CHANNEL_LABELS[ch].rstrip('*')}"]
            row += [_num(sub["eer"].get(s)) for s in seconds]
            row += [_pct(sub["detection_at_5pct_false_alarm"].get(s)) for s in (1, 30)]
            rows.append(row)
    _table_image(
        8, "Decisão por ligação (DEEP-VOICE, LOSO, augmentado): EER após N segundos de fala e detecção com 5% de alarme falso.",
        ["Modelo · canal", "EER 1s", "EER 3s", "EER 10s", "EER 30s", "Detecção 1s", "Detecção 30s"],
        rows, "tabela8_decisao_por_ligacao.png",
        col_widths=[0.34, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11], width_in=12,
    )


TABLES = [
    tabela1_datasets, tabela2_eer_principal, tabela3_asvspoof_detalhe, tabela4_teste_ao_vivo,
    tabela5_asvspoof2021, tabela6_portugues, tabela7_latencia, tabela8_decisao_por_ligacao,
]


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for table_fn in TABLES:
        try:
            table_fn()
        except FileNotFoundError as err:
            print(f"Pulando {table_fn.__name__}: falta {err.filename} (etapa ainda não rodada)")


if __name__ == "__main__":
    main()
