"""Fatiamento e preparação dos dados (src/build_dataset.py): nomes de arquivo,
remoção do atraso do codec, janelas de 1 s e reaproveitamento de janelas prontas.

Os codecs e a normalização de volume (que chamam o FFmpeg) são trocados por
versões falsas, então estes testes rodam em qualquer máquina."""
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src import build_dataset, channel_sim

SR = build_dataset.SR


class TestParseSpeakers:
    def test_real_recording_is_named_after_the_speaker(self):
        assert build_dataset.parse_speakers("Biden-original.wav", "REAL") == "biden"

    def test_fake_lists_both_identities_sorted(self):
        assert build_dataset.parse_speakers("obama-to-biden.wav", "FAKE") == "biden,obama"
        assert build_dataset.parse_speakers("biden-to-obama.wav", "FAKE") == "biden,obama"

    def test_fake_with_the_same_person_on_both_sides(self):
        assert build_dataset.parse_speakers("biden-to-biden.wav", "FAKE") == "biden"

    def test_unexpected_fake_name_is_rejected(self):
        with pytest.raises(ValueError):
            build_dataset.parse_speakers("gravacao.wav", "FAKE")


def test_file_seed_is_deterministic_and_depends_on_the_path():
    a, b = Path("data/raw/a.wav"), Path("data/raw/b.wav")
    assert build_dataset._file_seed(a) == build_dataset._file_seed(a)
    assert build_dataset._file_seed(a) != build_dataset._file_seed(b)
    assert 0 <= build_dataset._file_seed(a) < 2**32


class TestApplyCodecToRecording:
    @staticmethod
    def codec_with_delay(delay, tail=0):
        """Codec falso: devolve o áudio atrasado de `delay` amostras, com `tail` de sobra."""
        def codec(src, dst, seed):
            audio, sr = sf.read(src, dtype="float32")
            zeros = lambda n: np.zeros(n, dtype="float32")  # noqa: E731
            sf.write(dst, np.r_[zeros(delay), audio, zeros(tail)], sr)
        return codec

    def test_codec_delay_is_removed_and_the_length_is_preserved(self, tmp_path, monkeypatch, rng):
        audio = (0.1 * rng.standard_normal(2 * SR)).astype("float32")
        delay = channel_sim.CODEC_DELAY_SAMPLES["gsm"]
        monkeypatch.setitem(build_dataset.CODEC_FN, "gsm", self.codec_with_delay(delay, tail=100))
        out = build_dataset.apply_codec_to_recording(audio, "gsm", seed=1, tmp_dir=tmp_path)
        assert out.shape == audio.shape
        np.testing.assert_allclose(out, audio, atol=1e-3)

    def test_a_shorter_output_is_padded_with_silence(self, tmp_path, monkeypatch, rng):
        audio = (0.1 * rng.standard_normal(2 * SR)).astype("float32")

        def truncating_codec(src, dst, seed):
            data, sr = sf.read(src, dtype="float32")
            sf.write(dst, data[:-500], sr)

        monkeypatch.setitem(build_dataset.CODEC_FN, "vonr", truncating_codec)
        out = build_dataset.apply_codec_to_recording(audio, "vonr", seed=1, tmp_dir=tmp_path)
        assert out.shape == audio.shape
        assert np.all(out[-500:] == 0)


@pytest.fixture
def fake_pipeline(monkeypatch, rng):
    """Troca a leitura do áudio, o codec e a normalização de volume por versões falsas."""
    audio = (0.1 * rng.standard_normal(int(3.7 * SR))).astype("float32")
    calls = {"normalize": 0, "codec": []}

    monkeypatch.setattr(build_dataset, "load_trimmed", lambda path: audio)

    def fake_normalize(src, dst):
        calls["normalize"] += 1
        shutil.copyfile(src, dst)

    def fake_codec(a, channel, seed, tmp_dir):
        calls["codec"].append((channel, seed))
        return a * 0.5

    monkeypatch.setattr(channel_sim, "normalize_clean", fake_normalize)
    monkeypatch.setattr(build_dataset, "apply_codec_to_recording", fake_codec)
    return audio, calls


class TestSliceToWindows:
    def test_cuts_whole_one_second_windows(self, tmp_path, fake_pipeline):
        audio, calls = fake_pipeline
        paths = build_dataset.slice_to_windows(Path("raw/rec.wav"), tmp_path / "w")
        assert [p.name for p in paths] == ["rec_00000.wav", "rec_00001.wav", "rec_00002.wav"]  # 3,7 s -> 3 janelas
        for i, path in enumerate(paths):
            window, sr = sf.read(path, dtype="float32")
            assert sr == SR and len(window) == SR
            np.testing.assert_allclose(window, audio[i * SR:(i + 1) * SR], atol=1e-3)
        assert calls["normalize"] == 3
        assert calls["codec"] == []  # o canal limpo não passa por codec

    def test_reuses_finished_windows(self, tmp_path, fake_pipeline):
        _, calls = fake_pipeline
        first = build_dataset.slice_to_windows(Path("raw/rec.wav"), tmp_path / "w", expected=3)
        again = build_dataset.slice_to_windows(Path("raw/rec.wav"), tmp_path / "w", expected=3)
        assert again == first
        assert calls["normalize"] == 3  # a segunda chamada não refez nada

    def test_rebuilds_a_recording_that_was_interrupted(self, tmp_path, fake_pipeline):
        _, calls = fake_pipeline
        paths = build_dataset.slice_to_windows(Path("raw/rec.wav"), tmp_path / "w", expected=3)
        paths[2].unlink()  # simula uma execução interrompida no meio da gravação
        rebuilt = build_dataset.slice_to_windows(Path("raw/rec.wav"), tmp_path / "w", expected=3)
        assert len(rebuilt) == 3 and all(p.exists() for p in rebuilt)
        assert calls["normalize"] == 6

    def test_codec_channels_run_the_codec_once_on_the_whole_recording(self, tmp_path, fake_pipeline):
        audio, calls = fake_pipeline
        path = Path("raw/rec.wav")
        paths = build_dataset.slice_to_windows(path, tmp_path / "w", channel="gsm")
        assert calls["codec"] == [("gsm", build_dataset._file_seed(path))]
        window, _ = sf.read(paths[1], dtype="float32")
        np.testing.assert_allclose(window, 0.5 * audio[SR:2 * SR], atol=1e-3)

    def test_recordings_shorter_than_a_window_produce_nothing(self, tmp_path, monkeypatch, fake_pipeline):
        _, calls = fake_pipeline
        monkeypatch.setattr(build_dataset, "load_trimmed", lambda p: np.zeros(int(0.8 * SR), dtype="float32"))
        assert build_dataset.slice_to_windows(Path("raw/short.wav"), tmp_path / "w", channel="gsm") == []
        assert calls["codec"] == [] and calls["normalize"] == 0


def test_clean_windows_carry_the_recording_metadata(tmp_path, monkeypatch, fake_pipeline):
    raw = tmp_path / "raw"
    for label, names in (("REAL", ["biden-original.wav"]), ("FAKE", ["biden-to-obama.wav"])):
        (raw / label).mkdir(parents=True)
        for name in names:
            (raw / label / name).write_bytes(b"")
    monkeypatch.setattr(build_dataset, "RAW_DIR", raw)
    monkeypatch.setattr(build_dataset, "WINDOWS_DIR", tmp_path / "windows")

    rows, sources = build_dataset.build_clean_windows()

    assert set(sources) == {"biden-original.wav", "biden-to-obama.wav"}
    assert len(rows) == 6  # duas gravações de 3 janelas
    assert {r["label"] for r in rows} == {"real", "fake"}
    assert {r["speakers"] for r in rows} == {"biden", "biden,obama"}
    assert {r["channel"] for r in rows} == {"clean"}
    assert sorted(r["window_index"] for r in rows if r["label"] == "real") == [0, 1, 2]
