"""Treino (src/train.py): os três regimes de treino, rótulos, nomes de modelo e a
garantia central da avaliação: nenhum locutor de teste aparece no treino."""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src import train
from src.features import W2V_LAYER

ALL_CHANNELS = np.repeat(np.array(["clean", "gsm", "volte", "vonr", "mic", "vonr_loss"]), 10)
EVERYTHING = np.ones(len(ALL_CHANNELS), dtype=bool)


class TestRegimeMask:
    def test_clean_selects_only_the_clean_channel(self):
        mask = train.regime_mask("clean", ALL_CHANNELS, EVERYTHING)
        assert set(ALL_CHANNELS[mask]) == {"clean"} and mask.sum() == 10

    def test_augmented_uses_the_five_training_channels_and_never_the_test_only_one(self):
        mask = train.regime_mask("augmented", ALL_CHANNELS, EVERYTHING)
        assert set(ALL_CHANNELS[mask]) == set(train.TRAIN_CHANNELS)
        assert "vonr_loss" not in set(ALL_CHANNELS[mask])
        assert mask.sum() == 50  # cinco vezes mais janelas que o regime limpo

    def test_matched_control_has_as_many_windows_as_clean_but_from_the_augmented_pool(self):
        clean = train.regime_mask("clean", ALL_CHANNELS, EVERYTHING)
        augmented = train.regime_mask("augmented", ALL_CHANNELS, EVERYTHING)
        matched = train.regime_mask("augmented_matched", ALL_CHANNELS, EVERYTHING)
        assert matched.sum() == clean.sum()
        assert not (matched & ~augmented).any()
        assert len(set(ALL_CHANNELS[matched])) > 1  # vê canais diferentes, só que com menos dados

    def test_matched_control_is_reproducible(self):
        first = train.regime_mask("augmented_matched", ALL_CHANNELS, EVERYTHING)
        again = train.regime_mask("augmented_matched", ALL_CHANNELS, EVERYTHING)
        assert np.array_equal(first, again)

    @pytest.mark.parametrize("regime", train.REGIMES)
    def test_never_leaves_the_allowed_windows(self, regime):
        allowed = np.arange(len(ALL_CHANNELS)) % 2 == 0
        mask = train.regime_mask(regime, ALL_CHANNELS, allowed)
        assert not (mask & ~allowed).any()


def test_label_vector_marks_fake_as_one():
    df = pd.DataFrame({"label": ["real", "fake", "fake", "real"]})
    assert train.label_vector(df).tolist() == [0, 1, 1, 0]


class TestPredictionFrame:
    def test_keeps_the_metadata_columns_that_exist(self):
        df = pd.DataFrame({
            "channel": ["clean", "gsm", "clean"], "label": ["real", "fake", "fake"],
            "speakers": ["a", "b", "c"], "source_file": ["x", "y", "z"],
        })
        out = train.prediction_frame(df, np.array([True, False, True]), np.array([0.1, 0.9]),
                                     feature="mfcc", regime="clean")
        assert out["label"].tolist() == [0, 1]
        assert out["channel"].tolist() == ["clean", "clean"]
        assert out["proba"].tolist() == [0.1, 0.9]
        assert (out["feature"] == "mfcc").all() and (out["regime"] == "clean").all()
        assert {"speakers", "source_file"} <= set(out.columns)
        assert "window_index" not in out.columns  # não existia no dataset


class TestMakeClassifier:
    def test_mfcc_uses_a_balanced_random_forest(self):
        clf = train.make_classifier("mfcc")
        assert isinstance(clf, RandomForestClassifier)
        assert clf.class_weight == "balanced" and clf.random_state == train.RANDOM_SEED

    def test_wav2vec2_uses_standardized_logistic_regression(self):
        clf = train.make_classifier("wav2vec2")
        assert [name for name, _ in clf.steps] == ["standardscaler", "logisticregression"]
        assert isinstance(clf.steps[-1][1], LogisticRegression)
        assert clf.steps[-1][1].class_weight == "balanced"

    @pytest.mark.parametrize("feature", ["mfcc", "wav2vec2"])
    def test_learns_a_separable_problem(self, feature, rng):
        y = rng.integers(0, 2, 400)
        X = rng.normal(0, 1, (400, 6)) + 3 * y[:, None]
        clf = train.make_classifier(feature).fit(X[:300], y[:300])
        assert (clf.predict(X[300:]) == y[300:]).mean() > 0.95


class TestFinalModelPath:
    def test_default_layer_keeps_the_plain_name(self):
        assert train.final_model_path("wav2vec2", "clean", "asvspoof").name == "wav2vec2_clean_asvspoof.joblib"
        assert train.final_model_path("wav2vec2", "clean", "asvspoof", layer=W2V_LAYER).name == "wav2vec2_clean_asvspoof.joblib"

    def test_other_layers_get_the_layer_in_the_name(self):
        assert train.final_model_path("wav2vec2", "augmented", "asvspoof", layer=12).name == "wav2vec2-l12_augmented_asvspoof.joblib"

    def test_mfcc_ignores_the_layer(self):
        assert train.final_model_path("mfcc", "clean", "deepvoice", layer=12).name == "mfcc_clean_deepvoice.joblib"


def toy_speaker_frame():
    """Três locutores (a, b, c); as gravações falsas trazem a identidade de origem e de destino."""
    rows = []
    for channel in ("clean", "gsm"):
        for speaker in "abc":
            rows += [{"channel": channel, "label": "real", "speakers": speaker}] * 3
        for pair in ("a,b", "b,c", "a,c"):
            rows += [{"channel": channel, "label": "fake", "speakers": pair}] * 3
    return pd.DataFrame(rows)


class SpyClassifier:
    """Guarda quais linhas viram treino, em vez de aprender alguma coisa."""
    def __init__(self, fits):
        self.fits = fits

    def fit(self, X, y):
        self.fits.append(X[:, 0].astype(int))
        return self

    def predict_proba(self, X):
        return np.full((len(X), 2), 0.5)


class TestLeaveOneSpeakerOut:
    def test_the_held_out_speaker_never_reaches_the_training_set(self, monkeypatch):
        df = toy_speaker_frame()
        X = np.arange(len(df), dtype=float)[:, None]
        fits = []
        monkeypatch.setattr(train, "make_classifier", lambda feature: SpyClassifier(fits))

        result = train.run_loso(df, X, "mfcc")

        speakers = ["a", "b", "c"]
        assert len(fits) == len(speakers) * len(train.REGIMES)
        for k, rows in enumerate(fits):
            held_out = speakers[k // len(train.REGIMES)]
            assert len(rows) > 0
            for row in rows:
                assert held_out not in df.loc[row, "speakers"].split(",")
        for held_out in speakers:
            tested = result[result["held_out_speaker"] == held_out]
            # só entra no teste quem tem o locutor, seja como voz real, origem ou destino de conversão
            assert tested["speakers"].str.split(",").apply(lambda ids: held_out in ids).all()
            assert len(tested) == len(train.REGIMES) * 18


class TestOfficialSplit:
    def test_trains_only_on_the_train_split_and_tests_on_dev_and_eval(self, monkeypatch):
        rows = []
        for split in ("train", "dev", "eval"):
            for channel in ("clean", "gsm"):
                rows += [{"split": split, "channel": channel, "label": label, "speakers": f"{split}-{label}"}
                         for label in ("real", "fake") for _ in range(3)]
        df = pd.DataFrame(rows)
        X = np.arange(len(df), dtype=float)[:, None]
        fits = []
        monkeypatch.setattr(train, "make_classifier", lambda feature: SpyClassifier(fits))

        result = train.run_official_split(df, X, "wav2vec2")

        assert len(fits) == len(train.REGIMES)
        for rows_used in fits:
            assert set(df.loc[rows_used, "split"]) == {"train"}
        assert set(result["split"]) == {"dev", "eval"}
        assert set(result["regime"]) == set(train.REGIMES)
