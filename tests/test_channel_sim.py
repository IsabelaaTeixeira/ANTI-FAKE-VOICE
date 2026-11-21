"""Simulador de canais (src/channel_sim.py): as partes que não dependem dos
binários de codec, ou seja, a perda de pacotes em rajada, a escolha da taxa de
perda e as tabelas de constantes. A cadeia com FFmpeg e EVS é testada em
test_codecs.py."""
import numpy as np
import pytest

from src import channel_sim
from src.build_dataset import CODEC_FN


def write_g192_stream(path, n_frames=20000, payload_words=6, seed=0):
    """Bitstream G.192 sintético: por quadro, a palavra de sincronia, o número de
    palavras de carga e a carga."""
    rng = np.random.default_rng(seed)
    words = []
    for _ in range(n_frames):
        words += [channel_sim.G192_SYNC_GOOD, payload_words] + rng.integers(0, 2**16, payload_words).tolist()
    np.array(words, dtype="<u2").tofile(path)


def read_frames(path):
    """(sincronia, carga) de cada quadro do bitstream."""
    words = np.fromfile(path, dtype="<u2")
    frames, pos = [], 0
    while pos + 1 < len(words):
        size = int(words[pos + 1])
        frames.append((int(words[pos]), words[pos + 2: pos + 2 + size].copy()))
        pos += 2 + size
    return frames


def lost_flags(path):
    return [sync == channel_sim.G192_SYNC_BAD for sync, _ in read_frames(path)]


class TestDropG192Frames:
    def test_realized_loss_rate_matches_the_requested_rate(self, tmp_path):
        stream = tmp_path / "s.192"
        write_g192_stream(stream)
        rate = channel_sim.drop_g192_frames(stream, 0.06, np.random.default_rng(1))
        flags = lost_flags(stream)
        assert rate == pytest.approx(sum(flags) / len(flags))
        assert rate == pytest.approx(0.06, abs=0.012)

    def test_losses_come_in_bursts_of_about_two_frames(self, tmp_path):
        stream = tmp_path / "s.192"
        write_g192_stream(stream)
        channel_sim.drop_g192_frames(stream, 0.10, np.random.default_rng(2))
        runs, current = [], 0
        for lost in lost_flags(stream):
            if lost:
                current += 1
            elif current:
                runs.append(current)
                current = 0
        assert np.mean(runs) == pytest.approx(channel_sim.MEAN_BURST_FRAMES, abs=0.3)

    def test_only_the_sync_word_changes(self, tmp_path):
        stream = tmp_path / "s.192"
        write_g192_stream(stream, n_frames=2000)
        before = read_frames(stream)
        channel_sim.drop_g192_frames(stream, 0.1, np.random.default_rng(3))
        after = read_frames(stream)
        assert len(after) == len(before)
        assert {sync for sync, _ in after} <= {channel_sim.G192_SYNC_GOOD, channel_sim.G192_SYNC_BAD}
        assert any(sync == channel_sim.G192_SYNC_BAD for sync, _ in after)
        for (_, payload_before), (_, payload_after) in zip(before, after):
            assert np.array_equal(payload_before, payload_after)

    def test_zero_loss_rate_leaves_the_stream_untouched(self, tmp_path):
        stream = tmp_path / "s.192"
        write_g192_stream(stream, n_frames=500)
        original = stream.read_bytes()
        assert channel_sim.drop_g192_frames(stream, 0.0, np.random.default_rng(4)) == 0.0
        assert stream.read_bytes() == original

    def test_same_seed_gives_the_same_losses(self, tmp_path):
        outputs = []
        for name in ("a.192", "b.192"):
            stream = tmp_path / name
            write_g192_stream(stream, n_frames=2000)
            channel_sim.drop_g192_frames(stream, 0.1, np.random.default_rng(5))
            outputs.append(stream.read_bytes())
        assert outputs[0] == outputs[1]


class TestToVonrLoss:
    @staticmethod
    def record_calls(monkeypatch):
        calls = []
        monkeypatch.setattr(channel_sim, "to_vonr", lambda src, dst, fer=0.0, seed=None: calls.append((fer, seed)))
        return calls

    def test_loss_rate_is_drawn_from_the_documented_choices(self, tmp_path, monkeypatch):
        calls = self.record_calls(monkeypatch)
        for seed in range(60):
            channel_sim.to_vonr_loss(tmp_path / "in.wav", tmp_path / "out.wav", seed=seed)
        assert {fer for fer, _ in calls} == set(channel_sim.FER_CHOICES)

    def test_is_reproducible_for_a_given_seed(self, tmp_path, monkeypatch):
        calls = self.record_calls(monkeypatch)
        channel_sim.to_vonr_loss(tmp_path / "in.wav", tmp_path / "out.wav", seed=7)
        channel_sim.to_vonr_loss(tmp_path / "in.wav", tmp_path / "out.wav", seed=7)
        assert calls[0] == calls[1]


class TestConstants:
    def test_every_codec_channel_has_a_measured_delay(self):
        # build_dataset remove esse atraso antes de fatiar; um canal sem entrada
        # aqui quebraria o alinhamento das janelas entre canais.
        assert set(CODEC_FN) == set(channel_sim.CODEC_DELAY_SAMPLES)
        assert all(isinstance(d, int) and d >= 0 for d in channel_sim.CODEC_DELAY_SAMPLES.values())

    def test_loss_parameters_are_the_ones_the_paper_documents(self):
        # Taxa de perda sorteada entre 3, 6 e 10% por gravação, em rajadas de 2 quadros em média.
        assert channel_sim.FER_CHOICES == (0.03, 0.06, 0.10)
        assert channel_sim.MEAN_BURST_FRAMES == 2.0

    def test_vonr_uses_the_standard_evs_bitrate(self):
        assert channel_sim.EVS_BITRATE == "13200"


def test_tool_checks_report_missing_binaries(tmp_path, monkeypatch):
    monkeypatch.setattr(channel_sim, "FFMPEG", tmp_path / "no-ffmpeg")
    monkeypatch.setattr(channel_sim, "EVS_ENCODER", tmp_path / "EVS_cod")
    monkeypatch.setattr(channel_sim, "EVS_DECODER", tmp_path / "EVS_dec")
    assert channel_sim.check_amr_encoders_available() is False
    assert channel_sim.check_evs_available() is False
