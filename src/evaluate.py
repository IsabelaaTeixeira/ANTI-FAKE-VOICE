"""Calcula as métricas por canal x modelo x regime (e, no ASVspoof, por split)
a partir das predições de src/train.py, montando a tabela final da etapa 5.

Métricas:
  - EER (Equal Error Rate): o ponto onde a taxa de falso positivo (marcar voz
    real como clonada) se iguala à de falso negativo (deixar passar uma voz
    clonada). Não depende de limiar nem do desbalanceamento entre classes.
    Vem com intervalo de confiança de 95% por bootstrap AGRUPADO: reamostra
    locutores inteiros (no DEEP-VOICE, as dobras do LOSO), não janelas soltas
    — janelas do mesmo locutor são parecidas entre si, e tratá-las como
    independentes daria um intervalo falsamente estreito.
  - Acurácia balanceada: média entre a taxa de acerto em voz real e em voz
    clonada, no limiar 0,5. 50% = chute. Os dois datasets são ~90% fake, então
    a acurácia comum é enganosa (um modelo que diz "fake" pra tudo tira ~91%);
    ela continua na tabela só para comparação.
  - ASVspoof, extras: acurácia balanceada no split eval usando o limiar de EER
    calibrado no dev (como um sistema real escolheria o limiar, sem ver o
    teste); EER e min t-DCF por gravação (média das janelas de cada
    gravação). O t-DCF é a métrica oficial do ASVspoof 2019: o custo do
    detector quando ele protege um verificador de locutor (ASV), usando as
    pontuações ASV oficiais e os custos/probabilidades padrão do desafio.

Tabelas extras: EER por locutor (DEEP-VOICE) e por tipo de ataque (ASVspoof eval).

Uso:
    python -m src.evaluate
    python -m src.evaluate --dataset asvspoof
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from src.train import TRAIN_CHANNELS

DATASET_PATHS = {
    "deepvoice": {
        "pred": Path("results/cv_predictions.csv"),
        "table_out": Path("results/channel_model_table.csv"),
        "group_table_out": Path("results/speaker_table.csv"),
    },
    "asvspoof": {
        "pred": Path("results/cv_predictions_asvspoof.csv"),
        "table_out": Path("results/channel_model_table_asvspoof.csv"),
        "group_table_out": Path("results/attack_table_asvspoof.csv"),
    },
}
ASV_SCORES = {
    split: Path(f"data/raw/ASVSPOOF2019/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.{split}.gi.trl.scores.txt")
    for split in ("dev", "eval")
}
# Modelo de custo oficial do t-DCF do ASVspoof 2019.
TDCF_COST_MODEL = {
    "Pspoof": 0.05, "Ptar": (1 - 0.05) * 0.99, "Pnon": (1 - 0.05) * 0.01,
    "Cmiss_asv": 1, "Cfa_asv": 10, "Cmiss_cm": 1, "Cfa_cm": 10,
}
N_BOOTSTRAP = 1000
RANDOM_SEED = 42


def det_curve(target_scores: np.ndarray, nontarget_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Taxas de falsa rejeição (frr) e falsa aceitação (far) em todos os limiares
    possíveis — mesma implementação do script oficial do ASVspoof 2019."""
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))
    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, all_scores.size + 1) - tar_trial_sums)
    frr = np.concatenate(([0], tar_trial_sums / target_scores.size))
    far = np.concatenate(([1], nontarget_trial_sums / nontarget_scores.size))
    thresholds = np.concatenate(([all_scores[indices[0]] - 0.001], all_scores[indices]))
    return frr, far, thresholds


def eer_and_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """EER e o limiar onde ele acontece; y_score alto = mais provável ser fake (rótulo 1)."""
    frr, far, thresholds = det_curve(y_score[y_true == 1], y_score[y_true == 0])
    idx = int(np.argmin(np.abs(frr - far)))
    return float((frr[idx] + far[idx]) / 2), float(thresholds[idx])


def eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return eer_and_threshold(np.asarray(y_true), np.asarray(y_score))[0]


def bootstrap_eer_ci(y_true: np.ndarray, y_score: np.ndarray, clusters: np.ndarray) -> tuple[float, float]:
    """Intervalo de 95% do EER reamostrando grupos inteiros (locutores/dobras)."""
    rng = np.random.default_rng(RANDOM_SEED)
    groups = [np.flatnonzero(clusters == c) for c in np.unique(clusters)]
    values = []
    for _ in range(N_BOOTSTRAP):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if np.unique(y_true[idx]).size == 2:
            values.append(eer(y_true[idx], y_score[idx]))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def window_metrics(g: pd.DataFrame, cluster_col: str) -> dict:
    y, proba = g["label"].values, g["proba"].values
    pred = (proba >= 0.5).astype(int)
    ci_low, ci_high = bootstrap_eer_ci(y, proba, g[cluster_col].values)
    return {
        "n": len(g),
        "n_real": int((y == 0).sum()),
        "n_fake": int((y == 1).sum()),
        "accuracy": (pred == y).mean(),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "f1": f1_score(y, pred, pos_label=1, zero_division=0),
        "eer": eer(y, proba),
        "eer_ci_low": ci_low,
        "eer_ci_high": ci_high,
    }


def utterance_scores(g: pd.DataFrame) -> pd.DataFrame:
    """Uma pontuação por gravação: média do log-odds das janelas dela."""
    return (
        g.assign(logodds=logit(g["proba"].values))
        .groupby("source_file")
        .agg(label=("label", "first"), logodds=("logodds", "mean"))
    )


def asv_error_rates(split: str) -> tuple[float, float, float]:
    """Taxas de erro do verificador de locutor oficial, no limiar de EER dele."""
    asv = pd.read_csv(ASV_SCORES[split], sep=" ", header=None, names=["source", "kind", "score"])
    tar = asv.loc[asv["kind"] == "target", "score"].values
    non = asv.loc[asv["kind"] == "nontarget", "score"].values
    spoof = asv.loc[asv["kind"] == "spoof", "score"].values
    frr, far, thresholds = det_curve(tar, non)
    threshold = thresholds[int(np.argmin(np.abs(frr - far)))]
    return (non >= threshold).mean(), (tar < threshold).mean(), (spoof < threshold).mean()


def min_tdcf(bonafide_scores: np.ndarray, spoof_scores: np.ndarray, asv_rates: tuple[float, float, float]) -> float:
    """min t-DCF normalizado (ASVspoof 2019). Pontuações do detector: alto = bonafide."""
    pfa_asv, pmiss_asv, pmiss_spoof_asv = asv_rates
    cm = TDCF_COST_MODEL
    pmiss_cm, pfa_cm, _ = det_curve(bonafide_scores, spoof_scores)
    c1 = cm["Ptar"] * (cm["Cmiss_cm"] - cm["Cmiss_asv"] * pmiss_asv) - cm["Pnon"] * cm["Cfa_asv"] * pfa_asv
    c2 = cm["Cfa_cm"] * cm["Pspoof"] * (1 - pmiss_spoof_asv)
    return float(np.min((c1 * pmiss_cm + c2 * pfa_cm) / min(c1, c2)))


def asvspoof_extras(df: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    asv_rates = {split: asv_error_rates(split) for split in ("dev", "eval")}

    dev_thresholds = {}
    for (feature, regime), g in df[(df["split"] == "dev") & df["channel"].isin(TRAIN_CHANNELS)].groupby(["feature", "regime"]):
        dev_thresholds[(feature, regime)] = eer_and_threshold(g["label"].values, g["proba"].values)[1]

    extras = []
    for (split, feature, regime, channel), g in df.groupby(["split", "feature", "regime", "channel"]):
        utt = utterance_scores(g)
        pred_dev_threshold = (g["proba"].values >= dev_thresholds[(feature, regime)]).astype(int)
        extras.append({
            "split": split, "feature": feature, "regime": regime, "channel": channel,
            "balanced_accuracy_dev_threshold": balanced_accuracy_score(g["label"].values, pred_dev_threshold),
            "n_utterances": len(utt),
            "eer_utterance": eer(utt["label"].values, utt["logodds"].values),
            "min_tdcf": min_tdcf(
                -utt.loc[utt["label"] == 0, "logodds"].values,
                -utt.loc[utt["label"] == 1, "logodds"].values,
                asv_rates[split],
            ),
        })
    return table.merge(pd.DataFrame(extras), on=["split", "feature", "regime", "channel"])


def group_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """EER por locutor (DEEP-VOICE, dobra do LOSO) ou por tipo de ataque
    (ASVspoof eval, cada ataque contra todas as vozes reais do mesmo canal)."""
    rows = []
    for key, g in df.groupby(["feature", "regime", "channel"]):
        if group_col == "held_out_speaker":
            parts = {sp: sub for sp, sub in g.groupby("held_out_speaker")}
        else:
            g = g[g["split"] == "eval"]
            bonafide = g[g["label"] == 0]
            parts = {atk: pd.concat([bonafide, sub]) for atk, sub in g[g["label"] == 1].groupby("attack_id")}
        for name, sub in parts.items():
            rows.append({
                "feature": key[0], "regime": key[1], "channel": key[2], group_col: name,
                "n": len(sub), "eer": eer(sub["label"].values, sub["proba"].values),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["deepvoice", "asvspoof"], default="deepvoice")
    args = parser.parse_args()
    paths = DATASET_PATHS[args.dataset]

    df = pd.read_csv(paths["pred"])
    group_cols = ["feature", "regime", "channel"]
    if args.dataset == "asvspoof":
        group_cols = ["split"] + group_cols
        cluster_col = "speakers"
    else:
        cluster_col = "held_out_speaker"

    rows = []
    for key, g in df.groupby(group_cols):
        rows.append({**dict(zip(group_cols, key)), **window_metrics(g, cluster_col)})
    table = pd.DataFrame(rows)
    if args.dataset == "asvspoof":
        table = asvspoof_extras(df, table)
    table = table.sort_values(group_cols).reset_index(drop=True)

    paths["table_out"].parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(paths["table_out"], index=False)
    group_col = "attack_id" if args.dataset == "asvspoof" else "held_out_speaker"
    group_table(df, group_col).to_csv(paths["group_table_out"], index=False)

    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    pd.set_option("display.width", 250)
    print(table.to_string(index=False))
    print(f"\nTabela salva em {paths['table_out']} (e por {group_col} em {paths['group_table_out']})")


if __name__ == "__main__":
    main()
