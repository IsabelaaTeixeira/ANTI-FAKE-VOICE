"""Atributos de áudio (src/features.py): o vetor MFCC e as constantes do projeto."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import features


class TestMfccVector:
    def test_has_one_finite_value_per_declared_column(self, rng):
        y = (0.1 * rng.standard_normal(features.SR)).astype(np.float32)
        row = features.extract_mfcc_from_array(y)
        assert len(row) == len(features.MFCC_COLUMNS) == 6 + features.N_MFCC
        assert np.all(np.isfinite(row))

    def test_is_deterministic(self, rng):
        y = (0.1 * rng.standard_normal(features.SR)).astype(np.float32)
        assert features.extract_mfcc_from_array(y) == features.extract_mfcc_from_array(y)

    def test_noise_crosses_zero_more_often_than_a_low_tone(self, rng):
        t = np.arange(features.SR) / features.SR
        tone = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        noise = (0.1 * rng.standard_normal(features.SR)).astype(np.float32)
        zcr = features.MFCC_COLUMNS.index("zero_crossing_rate")
        assert features.extract_mfcc_from_array(noise)[zcr] > 5 * features.extract_mfcc_from_array(tone)[zcr]

    def test_rms_scales_with_the_amplitude(self, rng):
        y = (0.05 * rng.standard_normal(features.SR)).astype(np.float32)
        rms = features.MFCC_COLUMNS.index("rms")
        ratio = features.extract_mfcc_from_array(2 * y)[rms] / features.extract_mfcc_from_array(y)[rms]
        assert ratio == pytest.approx(2.0, rel=1e-3)

    def test_column_names_are_unique(self):
        assert len(set(features.MFCC_COLUMNS)) == len(features.MFCC_COLUMNS)


def test_meta_cols_for_keeps_only_existing_columns_in_canonical_order():
    df = pd.DataFrame(columns=["window_index", "label", "mfcc1", "filepath", "channel"])
    assert features.meta_cols_for(df) == ["filepath", "label", "channel", "window_index"]


class TestConstants:
    def test_layer_choices_are_valid_hidden_states(self):
        assert 0 <= features.W2V_LAYER < features.W2V_N_LAYERS
        assert features.W2V_ORIGINAL_LAYER == features.W2V_N_LAYERS - 1  # a última camada da rede
        assert features.W2V_LAYER != features.W2V_ORIGINAL_LAYER

    def test_every_dataset_has_the_four_output_paths(self):
        assert set(features.DATASET_PATHS) == {"deepvoice", "asvspoof", "asvspoof2021", "pt"}
        for paths in features.DATASET_PATHS.values():
            assert {"manifest", "mfcc_out", "w2v_meta_out", "w2v_layers_out"} <= set(paths)

    def test_stored_layers_file_sits_next_to_a_npy_or_inside_a_folder_of_blocks(self):
        assert features.stored_layers_path(Path("data/x.npy")) == Path("data/x.layers.json")
        assert features.stored_layers_path(Path("data/blocks")) == Path("data/blocks/layers.json")
