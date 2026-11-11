"""Testa a própria voz (real + clonada) contra os modelos finais JÁ
EXISTENTES, que nunca viram essa voz — o "antes" do retreino de
own_voice_ceiling.py.

Usa os modelos finais do regime augmented salvos por
`python -m src.train --save-final` (models/<atributo>_augmented_<dataset>.joblib)
e as mesmas janelas de own_voice_ceiling.build_own_voice_rows (mesmo
preparo do dataset, em todos os canais).

Uso:
    python -m src.own_voice_pretrain_test
    python -m src.own_voice_pretrain_test --dataset asvspoof
"""
import argparse

import joblib
import pandas as pd

from src.own_voice_ceiling import build_own_voice_rows, own_voice_features, summarize
from src.train import final_model_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["deepvoice", "asvspoof"], default="deepvoice")
    parser.add_argument("--features", choices=["mfcc", "wav2vec2", "both"], default="both")
    args = parser.parse_args()
    suffix = "" if args.dataset == "deepvoice" else f"_{args.dataset}"
    out_path = f"results/own_voice_pretrain_table{suffix}.csv"

    print("Fatiando e aplicando os canais na sua voz (real + clonada)...")
    own_df = build_own_voice_rows()

    rows = []
    for feature_name in ("mfcc", "wav2vec2"):
        if args.features not in (feature_name, "both"):
            continue
        clf = joblib.load(final_model_path(feature_name, "augmented", args.dataset))
        proba = clf.predict_proba(own_voice_features(own_df, feature_name))[:, 1]
        rows += summarize(own_df, proba, feature=feature_name, dataset=args.dataset)

    summary = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print("\nModelo original, sem retreino:")
    print(summary.to_string(index=False))
    summary.to_csv(out_path, index=False)
    print(f"\nTabela salva em {out_path}")


if __name__ == "__main__":
    main()
