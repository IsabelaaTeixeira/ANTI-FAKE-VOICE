"""Métricas de src/evaluate.py: EER, intervalo por bootstrap agrupado por locutor,
pontuação por gravação, min t-DCF e as tabelas por ataque e por locutor.

Os testes usam dados sintéticos com resposta conhecida (separação perfeita,
gaussianas com EER teórico, pontuações sem informação), então não dependem de
nenhum dataset."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src import evaluate
from src.evaluate import (
    asv_error_rates, bootstrap_eer_ci, det_curve, eer, eer_and_threshold,
    group_table, logit, min_tdcf, utterance_scores, window_metrics,
)


def separable(n_real=50, n_fake=50):
    """Duas classes sem sobreposição: voz real (0) com pontuação baixa, clonada (1) alta."""
    y = np.r_[np.zeros(n_real), np.ones(n_fake)].astype(int)
    score = np.r_[np.linspace(0.0, 0.4, n_real), np.linspace(0.6, 1.0, n_fake)]
    return y, score


def clustered(rng, n_clusters, per_cluster=20, shift=1.5, cluster_sd=0.5):
    """Janelas agrupadas por locutor. Cada locutor tem um deslocamento próprio de
    pontuação, então janelas do mesmo locutor não são independentes."""
    clusters = np.repeat(np.arange(n_clusters), per_cluster)
    y = np.tile(np.r_[np.zeros(per_cluster // 2), np.ones(per_cluster // 2)], n_clusters).astype(int)
    offsets = np.repeat(rng.normal(0, cluster_sd, n_clusters), per_cluster)
    score = rng.normal(0, 1, len(y)) + shift * y + offsets
    return y, score, clusters


class TestDetCurve:
    def test_shape_and_monotonicity(self):
        frr, far, thr = det_curve(np.array([0.7, 0.8, 0.9]), np.array([0.1, 0.2, 0.3, 0.75]))
        assert len(frr) == len(far) == len(thr) == 3 + 4 + 1
        assert (frr[0], far[0]) == (0, 1)
        assert (frr[-1], far[-1]) == (1, 0)
        assert np.all(np.diff(frr) >= 0)
        assert np.all(np.diff(far) <= 0)
        assert np.all(np.diff(thr) >= 0)


class TestEer:
    def test_perfect_separation_is_zero(self):
        y, score = separable()
        assert eer(y, score) == 0.0

    def test_inverted_scores_give_one(self):
        y, score = separable()
        assert eer(y, -score) == 1.0

    def test_uninformative_scores_give_one_half(self, rng):
        y = rng.integers(0, 2, 6000)
        score = rng.random(6000)
        assert eer(y, score) == pytest.approx(0.5, abs=0.04)

    def test_gaussian_case_matches_theory(self, rng):
        # Duas gaussianas de variância 1 a distância 2: EER teórico = Phi(-1) = 0,1587.
        n = 20000
        y = np.r_[np.zeros(n), np.ones(n)].astype(int)
        score = np.r_[rng.normal(0, 1, n), rng.normal(2, 1, n)]
        assert eer(y, score) == pytest.approx(norm.cdf(-1), abs=0.01)

    def test_threshold_sits_between_the_classes(self):
        y, score = separable()
        value, threshold = eer_and_threshold(y, score)
        assert value == 0.0
        assert score[y == 0].max() <= threshold < score[y == 1].min()

    def test_accepts_plain_lists(self):
        y, score = separable()
        assert eer(y.tolist(), score.tolist()) == 0.0


class TestBootstrap:
    def test_interval_contains_the_point_estimate(self, rng):
        y, score, clusters = clustered(rng, n_clusters=30)
        low, high = bootstrap_eer_ci(y, score, clusters)
        assert low <= eer(y, score) <= high

    def test_is_deterministic(self, rng):
        y, score, clusters = clustered(rng, n_clusters=20)
        assert bootstrap_eer_ci(y, score, clusters) == bootstrap_eer_ci(y, score, clusters)

    def test_more_speakers_give_a_narrower_interval(self):
        widths = []
        for n_clusters in (8, 80):
            y, score, clusters = clustered(np.random.default_rng(5), n_clusters)
            low, high = bootstrap_eer_ci(y, score, clusters)
            widths.append(high - low)
        assert widths[1] < widths[0]

    def test_clustered_interval_is_wider_than_treating_windows_as_independent(self):
        # A razão de reamostrar locutores inteiros (docstring de evaluate.py): com
        # locutores que diferem entre si, tratar cada janela como independente
        # daria um intervalo falsamente estreito.
        y, score, clusters = clustered(np.random.default_rng(7), n_clusters=25, per_cluster=40, cluster_sd=1.5)
        by_speaker = bootstrap_eer_ci(y, score, clusters)
        by_window = bootstrap_eer_ci(y, score, np.arange(len(y)))
        assert (by_speaker[1] - by_speaker[0]) > (by_window[1] - by_window[0])


class TestLogit:
    def test_clips_the_extremes(self):
        low, high = logit(np.array([0.0, 1.0]))
        assert low == pytest.approx(np.log(1e-4 / (1 - 1e-4)))
        assert high == pytest.approx(-low)

    def test_zero_at_one_half_and_monotonic(self):
        assert logit(np.array([0.5]))[0] == pytest.approx(0.0)
        assert np.all(np.diff(logit(np.linspace(0.001, 0.999, 50))) > 0)


class TestUtteranceScores:
    def test_one_score_per_recording_is_the_mean_log_odds(self):
        df = pd.DataFrame({
            "source_file": ["a", "a", "b", "b"], "label": [0, 0, 1, 1], "proba": [0.2, 0.4, 0.9, 0.7],
        })
        out = utterance_scores(df)
        assert list(out.index) == ["a", "b"]
        assert out["label"].tolist() == [0, 1]
        assert out.loc["a", "logodds"] == pytest.approx(logit(np.array([0.2, 0.4])).mean())
        assert out.loc["b", "logodds"] == pytest.approx(logit(np.array([0.9, 0.7])).mean())


class TestWindowMetrics:
    def test_perfect_detector(self):
        y, score = separable()
        g = pd.DataFrame({"label": y, "proba": score, "speakers": np.arange(len(y)) % 10})
        m = window_metrics(g, "speakers")
        assert (m["n"], m["n_real"], m["n_fake"]) == (100, 50, 50)
        assert m["accuracy"] == m["balanced_accuracy"] == m["f1"] == 1.0
        assert (m["eer"], m["eer_ci_low"], m["eer_ci_high"]) == (0.0, 0.0, 0.0)

    def test_plain_accuracy_is_misleading_with_unbalanced_classes(self, rng):
        # Um detector que diz "clonada" para tudo, com 90% de clonadas: a acurácia
        # comum dá 90%, a balanceada dá 50% (a razão de o projeto reportar EER).
        y = np.r_[np.zeros(100), np.ones(900)].astype(int)
        proba = rng.uniform(0.5, 1.0, 1000)
        g = pd.DataFrame({"label": y, "proba": proba, "speakers": np.arange(1000) % 10})
        m = window_metrics(g, "speakers")
        assert m["accuracy"] == pytest.approx(0.9)
        assert m["balanced_accuracy"] == pytest.approx(0.5)
        assert m["eer"] == pytest.approx(0.5, abs=0.08)


class TestMinTdcf:
    # (falso alarme do ASV, perda do ASV, perda do ASV para fala clonada)
    ASV = (0.01, 0.01, 0.5)

    def test_perfect_countermeasure_costs_nothing(self):
        bonafide = np.linspace(1, 2, 100)
        spoof = np.linspace(-2, -1, 100)
        assert min_tdcf(bonafide, spoof, self.ASV) == pytest.approx(0.0)

    def test_never_exceeds_the_trivial_system(self, rng):
        # Normalizado pelo melhor sistema trivial (aceitar tudo ou rejeitar tudo): <= 1.
        bonafide, spoof = rng.normal(size=2000), rng.normal(size=2000)
        assert 0.9 <= min_tdcf(bonafide, spoof, self.ASV) <= 1.0

    def test_improves_as_the_detector_improves(self, rng):
        spoof = rng.normal(0, 1, 3000)
        costs = [min_tdcf(rng.normal(shift, 1, 3000), spoof, self.ASV) for shift in (0.5, 2.0, 4.0)]
        assert costs[0] > costs[1] > costs[2]


class TestAsvErrorRates:
    def test_reads_the_official_score_file(self, tmp_path, monkeypatch, rng):
        n = 5000
        rows = []
        for kind, mean in (("target", 2.0), ("nontarget", -2.0), ("spoof", 0.0)):
            rows += [(f"LA_{kind}_{i}", kind, s) for i, s in enumerate(rng.normal(mean, 1, n))]
        path = tmp_path / "asv.txt"
        pd.DataFrame(rows).to_csv(path, sep=" ", header=False, index=False)
        monkeypatch.setitem(evaluate.ASV_SCORES, "dev", path)

        pfa, pmiss, pmiss_spoof = asv_error_rates("dev")
        # Alvos em N(2,1) e não-alvos em N(-2,1): o limiar de EER fica perto de 0,
        # e cada erro vale Phi(-2) = 0,023. A fala clonada em N(0,1) cai metade de cada lado.
        assert pfa == pytest.approx(norm.cdf(-2), abs=0.01)
        assert pmiss == pytest.approx(norm.cdf(-2), abs=0.01)
        assert pmiss_spoof == pytest.approx(0.5, abs=0.03)


class TestGroupTable:
    def test_eer_per_attack_on_the_eval_split(self):
        bonafide = pd.DataFrame({"label": 0, "proba": np.linspace(0.0, 0.4, 20), "attack_id": "-"})
        easy = pd.DataFrame({"label": 1, "proba": np.linspace(0.6, 1.0, 20), "attack_id": "A07"})
        hard = pd.DataFrame({"label": 1, "proba": np.linspace(-0.4, -0.0, 20), "attack_id": "A08"})
        df = pd.concat([bonafide, easy, hard], ignore_index=True).assign(
            feature="wav2vec2", regime="clean", channel="clean", split="eval",
        )
        table = group_table(df, "attack_id").set_index("attack_id")
        assert table.loc["A07", "eer"] == 0.0   # detecta o ataque sem erro
        assert table.loc["A08", "eer"] == 1.0   # o ataque que fica abaixo das vozes reais
        assert (table["n"] == 40).all()

    def test_eer_per_held_out_speaker(self):
        rows = []
        for speaker, flip in (("ana", 1), ("bia", -1)):
            real = np.linspace(0.0, 0.4, 10)
            fake = np.linspace(0.6, 1.0, 10)
            rows.append(pd.DataFrame({
                "label": [0] * 10 + [1] * 10,
                "proba": np.r_[real, fake] if flip > 0 else np.r_[fake, real],
                "held_out_speaker": speaker,
            }))
        df = pd.concat(rows, ignore_index=True).assign(feature="mfcc", regime="clean", channel="clean")
        table = group_table(df, "held_out_speaker").set_index("held_out_speaker")
        assert table.loc["ana", "eer"] == 0.0
        assert table.loc["bia", "eer"] == 1.0
