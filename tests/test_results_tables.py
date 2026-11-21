"""Tabelas de resultados versionadas em results/: consistência interna e os
números que o artigo e o README citam.

São testes de regressão sobre os CSVs do repositório (nenhum dataset é lido).
Se os experimentos forem refeitos e um número mudar de propósito, atualize o
teste e o texto do artigo juntos."""
from pathlib import Path

import pandas as pd
import pytest

from src.features import W2V_LAYER

RESULTS = Path(__file__).resolve().parent.parent / "results"


def table(name):
    return pd.read_csv(RESULTS / name)


def row(df, **conditions):
    """A única linha que satisfaz as condições."""
    mask = pd.Series(True, index=df.index)
    for column, value in conditions.items():
        mask &= df[column] == value
    found = df[mask]
    assert len(found) == 1, f"esperava 1 linha para {conditions}, achei {len(found)}"
    return found.iloc[0]


def tables_with_eer_interval():
    names = []
    for path in sorted(RESULTS.glob("*.csv")):
        header = set(pd.read_csv(path, nrows=0).columns)
        if {"eer", "eer_ci_low", "eer_ci_high"} <= header:
            names.append(path.name)
    return names


@pytest.mark.parametrize("name", tables_with_eer_interval())
def test_every_eer_is_a_rate_inside_its_own_confidence_interval(name):
    df = table(name)
    assert df["eer"].notna().all() and df["eer"].between(0, 1).all()
    # Algumas linhas (por exemplo, o EER por locutor conhecido) não têm intervalo;
    # onde ele existe, tem que conter a estimativa.
    with_interval = df.dropna(subset=["eer_ci_low", "eer_ci_high"])
    assert len(with_interval) > 0
    assert (with_interval["eer_ci_low"] <= with_interval["eer"] + 1e-12).all()
    assert (with_interval["eer"] <= with_interval["eer_ci_high"] + 1e-12).all()


class TestAsvspoof2019:
    def test_sequence_classifier_beats_the_mean_classifier_on_new_attacks(self):
        df = table("full_asvspoof2019.csv")
        seq = row(df, split="eval", channel="clean", model="seq_full_clean")
        mean = row(df, split="eval", channel="clean", model="lr_full_clean")
        assert seq["eer"] == pytest.approx(0.1156, abs=5e-4)           # 11,6% por janela
        assert seq["eer_utterance"] == pytest.approx(0.097, abs=1e-3)  # 9,7% por gravação
        assert seq["min_tdcf"] == pytest.approx(0.228, abs=1e-3)
        assert mean["eer"] == pytest.approx(0.149, abs=1e-3)
        assert seq["eer"] < mean["eer"]

    def test_ten_times_more_data_help_the_mean_classifier_only_a_little(self):
        df = table("full_asvspoof2019.csv")
        sample = row(df, split="eval", channel="clean", model="lr_sample_clean")["eer"]
        full = row(df, split="eval", channel="clean", model="lr_full_clean")["eer"]
        assert 0 < sample - full < 0.02

    def test_new_attacks_cost_more_than_new_speakers(self):
        df = table("full_asvspoof2019.csv")
        dev = row(df, split="dev", channel="clean", model="seq_full_clean")["eer"]
        evaluation = row(df, split="eval", channel="clean", model="seq_full_clean")["eer"]
        assert dev == pytest.approx(0.017, abs=2e-3)
        assert evaluation > 5 * dev


class TestAsvspoof2021:
    def test_training_with_channels_cuts_the_gsm_false_alarm(self):
        df = table("full_asvspoof2021.csv")
        clean = row(df, channel="gsm", model="seq_full_clean")
        augmented = row(df, channel="gsm", model="seq_full_augmented")
        assert clean["false_alarm_rate_utterance_dev_threshold"] == pytest.approx(0.291, abs=1e-3)
        assert augmented["false_alarm_rate_utterance_dev_threshold"] == pytest.approx(0.084, abs=1e-3)

    def test_the_landline_network_which_was_never_simulated_also_improves(self):
        df = table("full_asvspoof2021.csv")
        clean = row(df, channel="pstn", model="seq_full_clean")["eer_utterance"]
        augmented = row(df, channel="pstn", model="seq_full_augmented")["eer_utterance"]
        assert clean == pytest.approx(0.224, abs=1e-3)
        assert augmented == pytest.approx(0.170, abs=1e-3)

    def test_overall_false_alarm_drops_at_the_cost_of_some_detection(self):
        df = table("full_asvspoof2021.csv")
        clean = row(df, channel="todas", model="seq_full_clean")
        augmented = row(df, channel="todas", model="seq_full_augmented")
        assert clean["false_alarm_rate_utterance_dev_threshold"] == pytest.approx(0.073, abs=1e-3)
        assert augmented["false_alarm_rate_utterance_dev_threshold"] == pytest.approx(0.026, abs=1e-3)
        assert clean["detection_rate_utterance_dev_threshold"] == pytest.approx(0.691, abs=1e-3)
        assert augmented["detection_rate_utterance_dev_threshold"] == pytest.approx(0.651, abs=1e-3)

    def test_reference_detectors_were_evaluated_on_the_same_recordings(self):
        baselines = table("asvspoof2021_baselines.csv")
        overall = baselines[baselines["channel"] == "todas"].set_index("baseline")
        assert set(overall.index) == {"CQCC-GMM", "LFCC-GMM", "LFCC-LCNN", "RawNet2"}
        assert (overall["n_utterances"] == 148176).all()
        assert (overall["n_utterances_with_windows"] == 137496).all()
        assert overall.loc["LFCC-LCNN", "eer_utterance_with_windows"] < 0.10
        assert overall.loc["RawNet2", "eer_utterance_with_windows"] < 0.10


class TestCodecSimulationMode:
    """Codificar cada janela isolada, em vez da gravação inteira, superestima o efeito do canal."""

    def test_the_mean_classifier_is_unaffected_in_every_model(self):
        df = table("full_perwindow_paired.csv")
        lr = df[df["classifier"] == "lr"]
        assert len(lr) > 0
        assert (lr["diff_ci_low"] <= 0).all() and (lr["diff_ci_high"] >= 0).all()

    def test_the_sequence_classifier_overestimates_gsm_and_vonr(self):
        df = table("full_perwindow_paired.csv")
        for channel in ("gsm", "vonr"):
            result = row(df, model="seq_full_clean", channel=channel)
            assert result["diff_ci_low"] > 0
            assert result["diff"] == pytest.approx(0.026, abs=2e-3)


class TestLayerChoice:
    def test_the_layer_in_use_is_the_best_one_on_the_validation_split(self):
        sweep = table("layer_sweep.csv")
        dev = sweep[(sweep["dataset"] == "asvspoof") & (sweep["split"] == "dev")]
        training_channels = dev[dev["channel"].isin(["clean", "gsm", "volte", "vonr", "mic"])]
        mean_eer = training_channels.groupby("layer")["eer"].mean()
        assert int(mean_eer.idxmin()) == W2V_LAYER
        assert mean_eer[W2V_LAYER] == pytest.approx(0.093, abs=1e-3)
        assert mean_eer[12] == pytest.approx(0.139, abs=1e-3)


class TestLatency:
    def test_every_architecture_fits_in_the_real_time_budget(self):
        latency = table("latency.csv")
        assert latency["cabe_em_paralelo_1s"].all()
        assert latency["cabe_no_caminho_da_voz_150ms"].all()
        assert (latency["p95_ms"] >= latency["mediana_ms"]).all()

    def test_cutting_and_quantizing_the_network_makes_it_faster(self):
        latency = table("latency.csv").set_index("arquitetura")
        full = latency.loc["Wav2Vec2 completo (camada 12) + Regressão Logística", "mediana_ms"]
        cut = latency.loc["Wav2Vec2 cortado na camada 4 + Regressão Logística", "mediana_ms"]
        int8 = latency.loc["Wav2Vec2 cortado na camada 4, int8 + Regressão Logística", "mediana_ms"]
        assert full > cut > int8
        assert latency.loc["Wav2Vec2 cortado na camada 4, int8 + classificador de sequência", "mediana_ms"] < 40
