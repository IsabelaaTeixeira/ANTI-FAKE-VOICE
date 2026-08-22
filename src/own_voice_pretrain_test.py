"""Testa a própria voz (real + clonada) contra o modelo JÁ EXISTENTE
(models/mfcc_rf_final.joblib, que NUNCA viu essa voz) — mesma metodologia
de speaker_dependent_eval.py, mas para comparar o "antes" (sem retreinar)
com o resultado do teste de teto em own_voice_ceiling.py (com retreino).

Estende as fases M/N (que só cobriram o canal clean/ao vivo) para os 4
canais, usando o mesmo pipeline de fatiamento + simulação de canal.

Uso:
    python -m src.own_voice_pretrain_test
    python -m src.own_voice_pretrain_test --dataset asvspoof
"""
import argparse

import joblib
import pandas as pd
from sklearn.metrics import f1_score

from src.features import META_COLS
from src.own_voice_ceiling import build_own_voice_rows, eer

MODEL_PATHS = {
    "deepvoice": "models/mfcc_rf_final.joblib",
    "asvspoof": "models/mfcc_rf_final_asvspoof.joblib",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["deepvoice", "asvspoof"], default="deepvoice")
    args = parser.parse_args()
    model_path = MODEL_PATHS[args.dataset]
    out_path = f"results/own_voice_pretrain_table_{args.dataset}.csv" if args.dataset != "deepvoice" else "results/own_voice_pretrain_table.csv"

    print("Fatiando e aplicando os 4 canais na sua voz (real + clonada)...")
    own_df = build_own_voice_rows()

    feat_cols = [c for c in own_df.columns if c not in META_COLS]
    clf = joblib.load(model_path)

    own_eval = own_df.sort_values(["label", "channel", "filepath"]).reset_index(drop=True)
    proba_all = clf.predict_proba(own_eval[feat_cols].values)[:, 1]

    print(f"\n{'janela':>8} | {'canal':<6} | {'rótulo real':<12} | {'veredito':<8} | prob. de ser clonada")
    print("-" * 66)
    for i, (row, proba) in enumerate(zip(own_eval.itertuples(), proba_all), 1):
        veredito = "FAKE" if proba >= 0.5 else "real"
        barra = "█" * int(proba * 20)
        print(f"{i:>7}  | {row.channel:<6} | {row.label:<12} | {veredito:<8} | {proba:>5.1%} {barra}")

    print("\nResumo por canal (modelo original, sem retreino):")
    rows = []
    for channel, sub in own_eval.groupby("channel"):
        y = (sub["label"] == "fake").astype(int).values
        proba = clf.predict_proba(sub[feat_cols].values)[:, 1]
        pred = (proba >= 0.5).astype(int)
        rows.append({
            "channel": channel, "n": len(sub),
            "accuracy": (pred == y).mean(),
            "f1": f1_score(y, pred, pos_label=1, zero_division=0),
            "eer": eer(y, proba),
        })
    summary = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(summary.to_string(index=False))
    summary.to_csv(out_path, index=False)
    print(f"\nTabela salva em {out_path}")


if __name__ == "__main__":
    main()
