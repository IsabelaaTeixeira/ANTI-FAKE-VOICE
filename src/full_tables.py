"""Tabelas do artigo de congresso, a partir das predições de src/full_predict.py.

Cada coluna de predição é um modelo: <classificador>_<treino>_<regime>, com
classificador lr (média + Regressão Logística) ou seq (classificador de
sequência), treino sample (amostra de 2.683 gravações) ou full (split de
treino completo, 25.380 gravações) e regime clean / augmented /
augmented_matched.

Saídas (results/):
  full_asvspoof2019.csv   dev e eval do ASVspoof 2019, por canal simulado
  full_perwindow.csv      eval, GSM/VoLTE/VoNR: codec na gravação inteira x em cada janela isolada
  full_perwindow_paired.csv  a diferença de EER entre os dois modos, janela a janela, com IC pareado
  full_asvspoof2021.csv   ASVspoof 2021 LA (telefonia real), por condição
  full_pt.csv             conjunto em português, por canal e sintetizador

Nas duas tabelas externas, false_alarm_rate e detection_rate (de
src/external_eval.py) usam o limiar fixo 0,5; as colunas *_dev_threshold usam
o limiar de EER do dev do ASVspoof 2019, fixado antes do teste, por janela e
por gravação (média do log-odds das janelas contra o log-odds do limiar, os dois
com a mesma limitação de src.evaluate.logit).

Uma tabela já salva e mais nova que todas as predições de que depende é
reaproveitada (o bootstrap tem semente fixa, então recalcular daria o mesmo
resultado); --force recalcula todas.

Uso:
    python -m src.full_tables [--force]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import (
    N_BOOTSTRAP, RANDOM_SEED, asv_error_rates, eer, eer_and_threshold, logit, min_tdcf, utterance_scores, window_metrics,
)
from src.external_eval import metrics as external_metrics
from src.full_predict import OUT_DIR, load_predictions
from src.train import TRAIN_CHANNELS

RESULTS = Path("results")


def model_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(("lr_", "seq_"))]


def describe(col: str) -> dict:
    classifier, train_set, regime = col.split("_", 2)
    return {"model": col, "classifier": classifier, "train_set": train_set, "regime": regime}


def with_scores(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return pd.DataFrame({
        "label": (df["label"] == "fake").astype(int).values, "proba": df[col].values,
        "speakers": df["speakers"].values, "source_file": df["source_file"].values,
    })


def dev_thresholds(asv: pd.DataFrame) -> dict[str, float]:
    dev = asv[(asv["split"] == "dev") & asv["channel"].isin(TRAIN_CHANNELS)]
    y = (dev["label"] == "fake").astype(int).values
    return {col: eer_and_threshold(y, dev[col].values)[1] for col in model_columns(asv)}


def table_asvspoof2019(asv: pd.DataFrame) -> pd.DataFrame:
    asv_rates = {split: asv_error_rates(split) for split in ("dev", "eval")}
    rows = []
    for (split, channel), g in asv.groupby(["split", "channel"]):
        for col in model_columns(asv):
            s = with_scores(g, col)
            utt = utterance_scores(s)
            rows.append({
                "split": split, "channel": channel, **describe(col), **window_metrics(s, "speakers"),
                "eer_utterance": eer(utt["label"].values, utt["logodds"].values),
                "min_tdcf": min_tdcf(-utt.loc[utt["label"] == 0, "logodds"].values,
                                     -utt.loc[utt["label"] == 1, "logodds"].values, asv_rates[split]),
            })
    return pd.DataFrame(rows)


def table_perwindow(asv: pd.DataFrame, per_window: pd.DataFrame) -> pd.DataFrame:
    """Mesmo modelo, mesmas gravações e janelas: só muda a forma de simular o codec.
    A primeira janela de cada gravação é igual nos dois modos (o codec começa do
    zero no início da gravação em ambos), por isso também a comparação só com as
    janelas a partir da segunda, onde o efeito aparece por inteiro. O canal limpo
    entra como referência (simulation="continuous"), nas mesmas janelas."""
    channels = [*per_window["channel"].unique(), "clean"]
    continuous = asv[(asv["split"] == "eval") & asv["channel"].isin(channels)]
    both = pd.concat([continuous.assign(simulation="continuous"), per_window.assign(simulation="per_window")], ignore_index=True)
    rows = []
    for windows_kept, sub in (("todas", both), ("a partir da 2a", both[both["window_index"] >= 1])):
        for (simulation, channel), g in sub.groupby(["simulation", "channel"]):
            for col in model_columns(both):
                s = with_scores(g, col)
                m = window_metrics(s, "speakers")
                rows.append({"windows": windows_kept, "simulation": simulation, "channel": channel, **describe(col),
                             "n": m["n"], "eer": m["eer"], "eer_ci_low": m["eer_ci_low"], "eer_ci_high": m["eer_ci_high"]})
    return pd.DataFrame(rows)


def table_perwindow_paired(asv: pd.DataFrame, per_window: pd.DataFrame) -> pd.DataFrame:
    """EER(codec em cada janela) - EER(codec na gravação inteira), nas mesmas janelas
    (a partir da segunda), com intervalo de 95% por bootstrap pareado: cada
    reamostragem de locutores vale para os dois modos ao mesmo tempo."""
    keys = ["source_file", "window_index", "channel"]
    continuous = asv[(asv["split"] == "eval") & asv["channel"].isin(per_window["channel"].unique()) & (asv["window_index"] >= 1)]
    both = continuous.merge(per_window[per_window["window_index"] >= 1], on=keys, suffixes=("_c", "_p"))
    rows = []
    for channel, g in both.groupby("channel"):
        y = (g["label_c"] == "fake").astype(int).values
        speakers = g["speakers_c"].values
        groups = [np.flatnonzero(speakers == s) for s in np.unique(speakers)]
        for col in model_columns(asv):
            a, b = g[col + "_c"].values, g[col + "_p"].values
            rng = np.random.default_rng(RANDOM_SEED)
            diffs = []
            for _ in range(N_BOOTSTRAP):
                idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
                if np.unique(y[idx]).size == 2:
                    diffs.append(eer(y[idx], b[idx]) - eer(y[idx], a[idx]))
            rows.append({"channel": channel, **describe(col), "n": len(g),
                         "eer_continuous": eer(y, a), "eer_per_window": eer(y, b), "diff": eer(y, b) - eer(y, a),
                         "diff_ci_low": np.percentile(diffs, 2.5), "diff_ci_high": np.percentile(diffs, 97.5),
                         "score_corr": np.corrcoef(a, b)[0, 1]})
    return pd.DataFrame(rows)


def dev_threshold_rates(s: pd.DataFrame, threshold: float) -> dict:
    """Voz real marcada como clonada (alarme falso) e clones detectados, no limiar do dev."""
    y, called = s["label"].values == 1, s["proba"].values >= threshold
    utt = utterance_scores(s)
    uy, utt_called = utt["label"].values == 1, utt["logodds"].values >= logit(threshold)
    return {"false_alarm_rate_dev_threshold": called[~y].mean(), "detection_rate_dev_threshold": called[y].mean(),
            "false_alarm_rate_utterance_dev_threshold": utt_called[~uy].mean(),
            "detection_rate_utterance_dev_threshold": utt_called[uy].mean()}


def table_external(pred: pd.DataFrame, thresholds: dict[str, float], by_generator: bool) -> pd.DataFrame:
    rows = []
    for channel, g in [("todas", pred), *pred.groupby("channel")]:
        parts = {"todos": g}
        if by_generator:
            bonafide = g[g["label"] == "real"]
            for generator, fake in g[g["label"] == "fake"].groupby("generator"):
                parts[generator] = pd.concat([bonafide, fake])
        for generator, sub in parts.items():
            for col in model_columns(pred):
                s = with_scores(sub, col)
                rows.append({"channel": channel, "generator": generator, **describe(col),
                             **external_metrics(s, thresholds[col]), **dev_threshold_rates(s, thresholds[col])})
    return pd.DataFrame(rows)


def newest_prediction(source: str) -> float:
    return max(p.stat().st_mtime for p in (OUT_DIR / source).glob("part_*.csv"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="recalcula também as tabelas já atualizadas")
    force = parser.parse_args().force
    asv = load_predictions("asvspoof")
    thresholds = dev_thresholds(asv)
    outputs = {"full_asvspoof2019.csv": (["asvspoof"], lambda: table_asvspoof2019(asv))}
    if (OUT_DIR / "perwindow").exists():
        outputs["full_perwindow.csv"] = (["asvspoof", "perwindow"], lambda: table_perwindow(asv, load_predictions("perwindow")))
        outputs["full_perwindow_paired.csv"] = (["asvspoof", "perwindow"],
                                                lambda: table_perwindow_paired(asv, load_predictions("perwindow")))
    if (OUT_DIR / "asvspoof2021").exists():
        outputs["full_asvspoof2021.csv"] = (["asvspoof", "asvspoof2021"],
                                            lambda: table_external(load_predictions("asvspoof2021"), thresholds, by_generator=False))
    if (OUT_DIR / "pt").exists():
        outputs["full_pt.csv"] = (["asvspoof", "pt"], lambda: table_external(load_predictions("pt"), thresholds, by_generator=True))
    for name, (sources, build) in outputs.items():
        path = RESULTS / name
        if not force and path.exists() and path.stat().st_mtime > max(newest_prediction(s) for s in sources):
            print(f"{path} já está atualizada; mantida")
            continue
        table = build()
        table.to_csv(path, index=False)
        print(f"Salvo {path}: {len(table)} linhas")


if __name__ == "__main__":
    main()
