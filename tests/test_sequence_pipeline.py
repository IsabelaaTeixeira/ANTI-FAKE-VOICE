"""Atributos de sequência e armazenamento em blocos (src/sequence_features.py e
as partes de src/sequence_classifier.py que não dependem do PyTorch)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import sequence_classifier as sc
from src import sequence_features as sf_


def write_part(directory, index, channels, labels, pooled, seq):
    stem = directory / f"part_{index:03d}"
    pd.DataFrame({"channel": channels, "label": labels}).to_csv(f"{stem}.csv", index=False)
    np.save(f"{stem}.pooled.npy", pooled)
    np.save(f"{stem}.seq.npy", seq)


@pytest.fixture
def store_dir(tmp_path, rng):
    pooled = [rng.normal(size=(3, 4)).astype(np.float16), rng.normal(size=(2, 4)).astype(np.float16)]
    seq = [rng.normal(size=(3, 5, 6)).astype(np.float16), rng.normal(size=(2, 5, 6)).astype(np.float16)]
    write_part(tmp_path, 0, ["clean", "clean", "gsm"], ["real", "fake", "fake"], pooled[0], seq[0])
    write_part(tmp_path, 1, ["vonr_loss", "mic"], ["real", "fake"], pooled[1], seq[1])
    return tmp_path, np.concatenate(pooled), np.concatenate(seq)


class TestFeatureStore:
    def test_concatenates_the_blocks_in_order(self, store_dir):
        directory, _, _ = store_dir
        store = sc.FeatureStore(directory)
        assert len(store.meta) == 5
        assert store.meta["channel"].tolist() == ["clean", "clean", "gsm", "vonr_loss", "mic"]
        assert store.offsets.tolist() == [0, 3, 5]

    def test_gathers_rows_across_blocks_in_the_requested_order(self, store_dir):
        directory, pooled, seq = store_dir
        store = sc.FeatureStore(directory)
        idx = np.array([4, 0, 3, 1, 2])
        got_pooled, got_seq = store.pooled_rows(idx), store.seq_rows(idx)
        assert got_pooled.dtype == np.float32 and got_seq.dtype == np.float32
        np.testing.assert_array_equal(got_pooled, pooled[idx].astype(np.float32))
        np.testing.assert_array_equal(got_seq, seq[idx].astype(np.float32))

    def test_empty_selection(self, store_dir):
        directory, _, _ = store_dir
        store = sc.FeatureStore(directory)
        assert store.pooled_rows(np.array([], dtype=int)).shape == (0, 4)

    def test_complains_when_there_are_no_blocks(self, tmp_path):
        with pytest.raises(SystemExit):
            sc.FeatureStore(tmp_path)

    def test_regimes_pick_the_same_channels_as_training(self, store_dir):
        directory, _, _ = store_dir
        store = sc.FeatureStore(directory)
        assert sc.regime_indices(store, "clean").tolist() == [0, 1]
        assert sc.regime_indices(store, "augmented").tolist() == [0, 1, 2, 4]  # sem o vonr_loss, só de teste
        assert len(sc.regime_indices(store, "augmented_matched")) == 2          # tantas janelas quanto o limpo

    def test_dev_selection_follows_the_regime(self, store_dir):
        directory, _, _ = store_dir
        dev = sc.FeatureStore(directory)
        assert sc.dev_indices(dev, "clean").tolist() == [0, 1]
        assert sc.dev_indices(dev, "augmented").tolist() == [0, 1, 2, 4]


def test_model_paths_follow_the_documented_names():
    assert sc.lr_model_path("clean") == Path("models/wav2vec2_clean_asvspoof_full.joblib")
    assert sc.seq_model_path("augmented") == Path("models/seqnet_augmented_asvspoof_full.pt")


class TestFramePca:
    @staticmethod
    def low_rank_frames(rng, n=600):
        latent = rng.normal(size=(n, 16))
        mix = rng.normal(size=(16, 768))
        return (latent @ mix + 0.05 * rng.normal(size=(n, 768))).astype(np.float32)

    def test_fit_saves_and_loads_the_same_parameters(self, tmp_path, monkeypatch, rng):
        monkeypatch.setattr(sf_, "PCA_PATH", tmp_path / "models" / "pca.npz")
        params = sf_.fit_pca(self.low_rank_frames(rng))
        assert params["components"].shape == (sf_.PCA_DIM, 768)
        assert params["mean"].shape == (768,) and np.all(params["scale"] > 0)
        assert np.all(np.diff(params["explained_variance_ratio"]) <= 1e-6)  # ordenada do maior para o menor
        loaded = sf_.load_pca()
        for key in params:
            np.testing.assert_array_equal(loaded[key], params[key])

    def test_projection_shape_dtype_and_scale(self, tmp_path, monkeypatch, rng):
        monkeypatch.setattr(sf_, "PCA_PATH", tmp_path / "pca.npz")
        frames = self.low_rank_frames(rng)
        params = sf_.fit_pca(frames)
        projected = sf_.project(frames.reshape(12, 50, 768), params)
        assert projected.shape == (12, 50, sf_.PCA_DIM) and projected.dtype == np.float16
        top = projected.astype(np.float32).reshape(-1, sf_.PCA_DIM)[:, :8]
        np.testing.assert_allclose(top.mean(axis=0), 0, atol=0.05)   # centrada
        np.testing.assert_allclose(top.std(axis=0), 1, atol=0.05)    # variância ~1 por dimensão

    def test_missing_pca_file_has_a_helpful_message(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sf_, "PCA_PATH", tmp_path / "missing.npz")
        with pytest.raises(SystemExit):
            sf_.load_pca()

    def test_encoding_no_audio_returns_empty_arrays_with_the_right_shapes(self):
        pooled, seqs = sf_.encode_windows([], None, None, {})
        assert pooled.shape == (0, 768) and seqs.shape == (0, sf_.N_FRAMES, sf_.PCA_DIM)
