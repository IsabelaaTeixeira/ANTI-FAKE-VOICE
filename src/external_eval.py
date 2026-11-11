"""Testes externos: aplica os modelos finais, SEM nenhum retreino, a dois
conjuntos que nunca participaram do treino nem da escolha de nada no projeto.

  - asvspoof2021: ASVspoof 2021 LA eval — mesmos ataques/locutores do eval de
    2019, mas transmitidos por telefonia REAL (7 condições de codec/rede).
    Pergunta central: treinar com canais SIMULADOS (regime augmented) ajuda
    nos canais REAIS? Para dar escala, a tabela inclui o EER dos 4 baselines
    oficiais do desafio (pontuações fornecidas nas chaves, uma por gravação)
    nas mesmas condições.
  - pt: vozes em português (MLS) clonadas por XTTS v2 e YourTTS, em todos os
    canais simulados (ver src/build_dataset_pt.py) — outro idioma e ataques
    mais recentes. Também mede a decisão por ligação (src/call_level.py),
    já que essas gravações têm 10-20s.

Modelos: os finais de `python -m src.train --save-final`, treinados no
ASVspoof 2019 (split de treino) e no DEEP-VOICE (todos os locutores), nos
três regimes. Além do Wav2Vec2 da camada escolhida (W2V_LAYER), entra também
o da camada da primeira versão (W2V_ORIGINAL_LAYER, linha "wav2vec2_l12" da
tabela; modelos de `python -m src.train --features wav2vec2 --layer 12
--final-only`): a camada foi escolhida no dev do ASVspoof 2019, e os testes
externos mostram se essa escolha vale também fora dele. Para os modelos do ASVspoof, a acurácia balanceada também é
medida no limiar de EER calibrado no dev de 2019 (como um sistema real
fixaria o limiar antes de ver dados novos).

Uso:
    python -m src.external_eval
    python -m src.external_eval --test-datasets pt
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from src.call_level import call_level_table
from src.evaluate import bootstrap_eer_ci, eer, eer_and_threshold, logit
from src.features import W2V_LAYER, W2V_ORIGINAL_LAYER, load_mfcc, load_wav2vec2
from src.train import PRED_PATHS, REGIMES, TRAIN_CHANNELS, final_model_path, label_vector

TABLE_OUT = Path("results/external_eval_table.csv")
BASELINES_OUT = Path("results/asvspoof2021_baselines.csv")
CALL_LEVEL_PT_OUT = Path("results/call_level_pt.csv")
BASELINES_DIR = Path("data/raw/ASVSPOOF2021/keys/LA/CM")
BASELINES = ("CQCC-GMM", "LFCC-GMM", "LFCC-LCNN", "RawNet2")
TRAIN_DATASETS = ("asvspoof", "deepvoice")
TEST_DATASETS = ("asvspoof2021", "pt")


def feature_variants() -> list[tuple[str, str, int | None]]:
    """(nome na tabela, atributo, camada do Wav2Vec2) de cada modelo avaliado."""
    variants = [("mfcc", "mfcc", None), ("wav2vec2", "wav2vec2", W2V_LAYER)]
    if W2V_ORIGINAL_LAYER != W2V_LAYER:
        variants.append((f"wav2vec2_l{W2V_ORIGINAL_LAYER}", "wav2vec2", W2V_ORIGINAL_LAYER))
    return variants


def dev_thresholds() -> dict[tuple[str, str], float]:
    """Limiar de EER no dev do ASVspoof 2019 (canais de treino), por atributo x regime.

    Para MFCC e Wav2Vec2 da camada escolhida, vem das predições de src.train.
    Para a camada de comparação (que src.train não avalia), aplica o modelo
    final do ASVspoof ao dev: dá o mesmo resultado, pois o modelo final do
    ASVspoof é treinado só com o split de treino, igual ao da avaliação."""
    pred = pd.read_csv(PRED_PATHS["asvspoof"])
    dev = pred[(pred["split"] == "dev") & pred["channel"].isin(TRAIN_CHANNELS)]
    thresholds = {key: eer_and_threshold(g["label"].values, g["proba"].values)[1] for key, g in dev.groupby(["feature", "regime"])}
    for name, feature_name, layer in feature_variants():
        if any(key[0] == name for key in thresholds):
            continue
        meta, X = load_wav2vec2("asvspoof", layer)
        is_dev = (meta["split"].values == "dev") & meta["channel"].isin(TRAIN_CHANNELS).values
        y, X_dev = label_vector(meta)[is_dev], X[is_dev]
        del X
        for regime in REGIMES:
            clf = joblib.load(final_model_path(feature_name, regime, "asvspoof", layer))
            thresholds[(name, regime)] = eer_and_threshold(y, clf.predict_proba(X_dev)[:, 1])[1]
    return thresholds


def metrics(df: pd.DataFrame, threshold: float | None) -> dict:
    y, proba = df["label"].values, df["proba"].values
    utt = df.assign(logodds=logit(proba)).groupby("source_file").agg(label=("label", "first"), logodds=("logodds", "mean"))
    ci_low, ci_high = bootstrap_eer_ci(y, proba, df["speakers"].values)
    called_fake = proba >= 0.5
    return {
        "n": len(df), "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum()),
        "balanced_accuracy": balanced_accuracy_score(y, called_fake.astype(int)),
        # as duas metades da acurácia balanceada, separadas: ela pode ficar em 0,5
        # tanto com um detector que não separa nada quanto com um que chama de
        # falso QUASE TUDO (reais inclusive) — esses dois números distinguem os casos
        "false_alarm_rate": called_fake[y == 0].mean(),
        "detection_rate": called_fake[y == 1].mean(),
        "balanced_accuracy_dev_threshold": (
            balanced_accuracy_score(y, (proba >= threshold).astype(int)) if threshold is not None else np.nan
        ),
        "eer": eer(y, proba), "eer_ci_low": ci_low, "eer_ci_high": ci_high,
        "eer_utterance": eer(utt["label"].values, utt["logodds"].values),
    }


def iter_model_predictions(test_dataset: str):
    """Predições de cada modelo final no conjunto externo, um modelo por vez
    (juntar as de todos numa tabela só não caberia na memória no ASVspoof 2021)."""
    for name, feature_name, layer in feature_variants():
        meta, X = load_mfcc(test_dataset) if feature_name == "mfcc" else load_wav2vec2(test_dataset, layer)
        meta = meta.assign(label=label_vector(meta)).reset_index(drop=True)
        for train_dataset in TRAIN_DATASETS:
            for regime in REGIMES:
                clf = joblib.load(final_model_path(feature_name, regime, train_dataset, layer))
                fixed = {"train_dataset": train_dataset, "feature": name, "regime": regime}
                yield fixed, meta.assign(proba=clf.predict_proba(X)[:, 1])
        del X


def evaluate_test_dataset(test_dataset: str, thresholds: dict) -> pd.DataFrame:
    rows, call_tables = [], []
    for fixed, pred in iter_model_predictions(test_dataset):
        threshold = thresholds.get((fixed["feature"], fixed["regime"])) if fixed["train_dataset"] == "asvspoof" else None
        # cada canal e todos juntos; no português, também cada sintetizador contra todas as vozes reais do canal
        for channel, g in [("todas", pred), *pred.groupby("channel")]:
            parts = {"todos": g}
            if test_dataset == "pt":
                bonafide = g[g["label"] == 0]
                for generator, fake in g[g["label"] == 1].groupby("generator"):
                    parts[generator] = pd.concat([bonafide, fake])
            for generator, sub in parts.items():
                rows.append({
                    "test_dataset": test_dataset, **fixed, "channel": channel, "generator": generator,
                    **metrics(sub, threshold),
                })
        if test_dataset == "pt":
            table = call_level_table(pred.assign(feature=fixed["feature"], regime=fixed["regime"]), "speakers", durations=(1, 2, 3, 5, 10))
            call_tables.append(table.assign(train_dataset=fixed["train_dataset"]))

    if call_tables:
        pd.concat(call_tables).to_csv(CALL_LEVEL_PT_OUT, index=False)
        print(f"Decisão por ligação (português) salva em {CALL_LEVEL_PT_OUT}")
    return pd.DataFrame(rows)


def baselines_2021() -> pd.DataFrame:
    """EER (por gravação) dos baselines oficiais do ASVspoof 2021 LA, por condição.

    Duas versões: em todas as gravações do eval (o número oficial) e só nas
    gravações que têm atributo neste projeto. Uma gravação com menos de 1s de
    fala depois do corte de silêncio não gera nenhuma janela (7,2% do eval, na
    mesma proporção entre real e falsa) e não pode ser pontuada pelos nossos
    modelos; a segunda versão compara os baselines com os nossos modelos
    exatamente nas mesmas gravações."""
    meta = pd.read_csv(
        BASELINES_DIR / "trial_metadata.txt", sep=" ", header=None,
        names=["speaker", "trial", "codec", "transmission", "attack_id", "key", "trim", "subset"],
    )
    meta = meta[meta["subset"] == "eval"]
    y = (meta["key"] == "spoof").astype(int).values
    with_windows = set(load_mfcc("asvspoof2021")[0]["source_file"].unique())
    rows = []
    for name in BASELINES:
        scores = pd.read_csv(BASELINES_DIR / name / "score.txt", sep=" ", header=None, names=["trial", "score"])
        merged = meta.assign(label=y).merge(scores, on="trial")
        for codec, g in [("todas", merged), *merged.groupby("codec")]:
            same = g[g["trial"].isin(with_windows)]
            # pontuação oficial: alto = bonafide; aqui, alto = fake
            rows.append({
                "baseline": name, "channel": codec,
                "n_utterances": len(g), "eer_utterance": eer(g["label"].values, -g["score"].values),
                "n_utterances_with_windows": len(same),
                "eer_utterance_with_windows": eer(same["label"].values, -same["score"].values),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-datasets", nargs="+", choices=TEST_DATASETS, default=list(TEST_DATASETS))
    args = parser.parse_args()

    thresholds = dev_thresholds()
    tables = [evaluate_test_dataset(ds, thresholds) for ds in args.test_datasets]
    table = pd.concat(tables, ignore_index=True)
    if TABLE_OUT.exists():
        previous = pd.read_csv(TABLE_OUT)
        table = pd.concat([previous[~previous["test_dataset"].isin(args.test_datasets)], table], ignore_index=True)
    table.to_csv(TABLE_OUT, index=False)

    if "asvspoof2021" in args.test_datasets:
        baselines_2021().to_csv(BASELINES_OUT, index=False)
        print(f"Baselines oficiais salvos em {BASELINES_OUT}")

    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    pd.set_option("display.width", 250)
    view = table[table["generator"] == "todos"]
    print(view.pivot_table(index=["test_dataset", "train_dataset", "feature", "regime"], columns="channel", values="eer").to_string())
    print(f"\nTabela salva em {TABLE_OUT}")


if __name__ == "__main__":
    main()
