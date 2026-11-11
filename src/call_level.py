"""Decisão por ligação: quanto o erro cai quando o detector acumula evidência
ao longo de vários segundos, em vez de julgar cada janela de 1s isolada.

Uma operadora não precisa saber se UM segundo é falso, e sim se a ligação é.
Aqui cada gravação é dividida em trechos consecutivos de N segundos (N
janelas seguidas, sem sobreposição) — cada trecho faz o papel dos "primeiros
N segundos de uma ligação". A pontuação do trecho é a média do log-odds das
janelas dele, e o EER é medido entre trechos, para N = 1, 2, 3, 5, 10, 20, 30.

Além do EER, reporta a taxa de detecção de voz clonada com 5% de alarme
falso (5 em cada 100 ligações legítimas marcadas como suspeitas) — o ponto
de operação mais natural para uma operadora.

Usa as predições LOSO do DEEP-VOICE (gravações longas, de minutos). No
ASVspoof as gravações têm poucos segundos, então a agregação por gravação
inteira já aparece em evaluate.py (eer_utterance).

Uso:
    python -m src.call_level
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from src.evaluate import bootstrap_eer_ci, eer, logit

PRED_PATH = Path("results/cv_predictions.csv")
OUT_PATH = Path("results/call_level_table.csv")
DURATIONS_S = (1, 2, 3, 5, 10, 20, 30)
FALSE_ALARM_RATE = 0.05


def chunk_scores(df: pd.DataFrame, seconds: int, keys: list[str]) -> pd.DataFrame:
    """Um log-odds médio por trecho de `seconds` janelas consecutivas."""
    df = df.sort_values(keys + ["source_file", "window_index"])
    df = df.assign(
        logodds=logit(df["proba"].values),
        chunk=df.groupby(keys + ["source_file"]).cumcount() // seconds,
    )
    grouped = df.groupby(keys + ["source_file", "chunk"])
    chunks = grouped.agg(label=("label", "first"), logodds=("logodds", "mean"), size=("logodds", "size"))
    return chunks[chunks["size"] == seconds].reset_index()


def detection_at_false_alarm(y: np.ndarray, score: np.ndarray, far: float) -> float:
    fpr, tpr, _ = roc_curve(y, score)
    return float(np.interp(far, fpr, tpr))


def call_level_table(df: pd.DataFrame, cluster_col: str, durations=DURATIONS_S) -> pd.DataFrame:
    keys = ["feature", "regime", "channel", cluster_col]
    rows = []
    for seconds in durations:
        chunks = chunk_scores(df, seconds, keys)
        for (feature, regime, channel), g in chunks.groupby(["feature", "regime", "channel"]):
            y, score = g["label"].values, g["logodds"].values
            if np.unique(y).size < 2:
                continue
            ci_low, ci_high = bootstrap_eer_ci(y, score, g[cluster_col].values)
            rows.append({
                "feature": feature, "regime": regime, "channel": channel, "seconds": seconds,
                "n_calls_real": int((y == 0).sum()), "n_calls_fake": int((y == 1).sum()),
                "eer": eer(y, score), "eer_ci_low": ci_low, "eer_ci_high": ci_high,
                "detection_at_5pct_false_alarm": detection_at_false_alarm(y, score, FALSE_ALARM_RATE),
            })
    return pd.DataFrame(rows)


def main() -> None:
    table = call_level_table(pd.read_csv(PRED_PATH), "held_out_speaker")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_PATH, index=False)

    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    pd.set_option("display.width", 200)
    view = table[table["regime"] == "augmented"].pivot_table(index=["feature", "channel"], columns="seconds", values="eer")
    print("EER por duração da ligação (regime augmented):")
    print(view.to_string())
    print(f"\nTabela salva em {OUT_PATH}")


if __name__ == "__main__":
    main()
