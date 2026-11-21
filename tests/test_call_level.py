"""Decisão por ligação (src/call_level.py): agregar segundos de evidência."""
import numpy as np
import pandas as pd
import pytest

from src.call_level import call_level_table, chunk_scores, detection_at_false_alarm
from src.evaluate import logit

KEYS = ["feature", "regime", "channel", "held_out_speaker"]


def one_recording(proba, label=1, source_file="a.wav"):
    return pd.DataFrame({
        "feature": "wav2vec2", "regime": "clean", "channel": "clean", "held_out_speaker": "ana",
        "source_file": source_file, "window_index": range(len(proba)), "label": label, "proba": proba,
    })


class TestChunkScores:
    def test_groups_consecutive_windows_and_drops_the_incomplete_tail(self):
        proba = [0.1, 0.3, 0.5, 0.7, 0.9]
        chunks = chunk_scores(one_recording(proba), seconds=2, keys=KEYS)
        assert len(chunks) == 2  # a quinta janela sozinha não forma um trecho de 2 s
        expected = [logit(np.array(proba[:2])).mean(), logit(np.array(proba[2:4])).mean()]
        assert chunks["logodds"].tolist() == pytest.approx(expected)

    def test_does_not_mix_recordings(self):
        df = pd.concat([
            one_recording([0.2, 0.2], label=0, source_file="real.wav"),
            one_recording([0.8, 0.8], label=1, source_file="fake.wav"),
        ], ignore_index=True)
        chunks = chunk_scores(df, seconds=2, keys=KEYS).sort_values("source_file")
        assert chunks["label"].tolist() == [1, 0]
        assert chunks["logodds"].iloc[0] > 0 > chunks["logodds"].iloc[1]

    def test_sorts_windows_before_chunking(self):
        shuffled = one_recording([0.1, 0.2, 0.8, 0.9]).sample(frac=1, random_state=0)
        ordered = chunk_scores(one_recording([0.1, 0.2, 0.8, 0.9]), seconds=2, keys=KEYS)
        assert chunk_scores(shuffled, seconds=2, keys=KEYS)["logodds"].tolist() == ordered["logodds"].tolist()


class TestDetectionAtFalseAlarm:
    def test_perfect_detector_catches_everything(self):
        y = np.r_[np.zeros(100), np.ones(100)]
        score = np.r_[np.linspace(0, 1, 100), np.linspace(2, 3, 100)]
        assert detection_at_false_alarm(y, score, 0.05) == pytest.approx(1.0)

    def test_useless_detector_detects_as_much_as_it_false_alarms(self, rng):
        y = rng.integers(0, 2, 6000)
        score = rng.random(6000)
        assert detection_at_false_alarm(y, score, 0.05) == pytest.approx(0.05, abs=0.03)


class TestCallLevelTable:
    def test_more_seconds_of_evidence_lower_the_error(self):
        # Cada janela é fraca (separa pouco), mas as janelas de uma gravação somam
        # evidência: o EER por trecho cai quando o trecho tem mais segundos.
        rng = np.random.default_rng(3)
        frames = []
        for i in range(40):
            label = i % 2
            logits = rng.normal(0.8 * (2 * label - 1), 2.0, 30)
            frames.append(one_recording(1 / (1 + np.exp(-logits)), label=label, source_file=f"f{i}.wav")
                          .assign(held_out_speaker=f"s{i % 8}"))
        table = call_level_table(pd.concat(frames, ignore_index=True), "held_out_speaker", durations=(1, 10))
        by_seconds = table.set_index("seconds")
        assert by_seconds.loc[10, "eer"] < by_seconds.loc[1, "eer"]
        assert by_seconds.loc[10, "detection_at_5pct_false_alarm"] >= by_seconds.loc[1, "detection_at_5pct_false_alarm"]
        assert {"eer_ci_low", "eer_ci_high", "n_calls_real", "n_calls_fake"} <= set(table.columns)
