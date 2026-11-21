"""Integração com a cadeia real de codecs: FFmpeg com AMR-NB e AMR-WB, e o codec
de referência EVS da 3GPP.

Só roda onde os binários existem: na imagem Docker (veja o Dockerfile) ou numa
máquina com tools/ montado como descreve o README. Fora disso, é pulado."""
import numpy as np
import pytest
import soundfile as sf
from scipy import signal

from src import channel_sim

pytestmark = [
    pytest.mark.codecs,
    pytest.mark.skipif(
        not (channel_sim.check_amr_encoders_available() and channel_sim.check_evs_available()),
        reason="FFmpeg com AMR e codec EVS não encontrados em tools/ (veja o Dockerfile)",
    ),
]

SR = 16000
CODECS = {"gsm": channel_sim.to_gsm, "volte": channel_sim.to_volte, "vonr": channel_sim.to_vonr}


def voiced_like(seconds=4.0, f0=120.0):
    """Sinal parecido com fala vozeada: trem de pulsos glóticos com altura variando
    devagar, passado por três ressonâncias (formantes) e com envoltória lenta."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    phase = 2 * np.pi * np.cumsum(f0 * (1 + 0.03 * np.sin(2 * np.pi * 0.7 * t)) / SR)
    x = np.maximum(0, np.sin(phase)) ** 8
    for freq, bandwidth in ((500, 80), (1500, 120), (2500, 160)):
        r = np.exp(-np.pi * bandwidth / SR)
        theta = 2 * np.pi * freq / SR
        x = signal.lfilter([1 - r], [1, -2 * r * np.cos(theta), r * r], x)
    x = x * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t) ** 2)
    return (0.3 * x / np.max(np.abs(x))).astype(np.float32)


def white_noise(seconds=3.0, seed=0):
    return (0.2 * np.random.default_rng(seed).standard_normal(int(seconds * SR))).astype(np.float32)


def through(codec, tmp_path, audio, **kwargs):
    src_path, dst_path = tmp_path / "in.wav", tmp_path / "out.wav"
    sf.write(src_path, audio, SR)
    codec(src_path, dst_path, **kwargs)
    out, sr = sf.read(dst_path, dtype="float32")
    assert sr == SR and out.ndim == 1
    return out


def band_ratio(x, low_hz, high_hz=SR / 2):
    freqs, power = signal.welch(x, SR, nperseg=1024)
    return power[(freqs >= low_hz) & (freqs < high_hz)].sum() / power.sum()


def delay_in_samples(original, coded):
    """Atraso do sinal codificado em relação ao original, pelo pico da correlação cruzada."""
    correlation = signal.correlate(coded, original, mode="full")
    return int(np.argmax(correlation)) - (len(original) - 1)


def test_toolchain_reports_ready():
    assert channel_sim.check_amr_encoders_available()
    assert channel_sim.check_evs_available()


@pytest.mark.parametrize("name", CODECS)
def test_every_codec_returns_16k_mono_audio_that_is_not_silent(name, tmp_path):
    out = through(CODECS[name], tmp_path, voiced_like())
    assert np.sqrt(np.mean(out ** 2)) > 0.01
    assert abs(len(out) - len(voiced_like())) <= 640  # no máximo dois quadros de 20 ms


def test_gsm_is_narrowband_and_the_wideband_codecs_are_not(tmp_path):
    noise = white_noise()
    ratios = {name: band_ratio(through(CODECS[name], tmp_path, noise), 4200, 6000) for name in CODECS}
    assert band_ratio(noise, 4200, 6000) > 0.2      # a entrada tem energia acima de 4 kHz
    assert ratios["gsm"] < 0.005                    # AMR-NB corta em ~4 kHz
    assert ratios["volte"] > 0.1                    # AMR-WB preserva até ~6,4 kHz
    assert ratios["vonr"] > 0.1                     # EVS em banda larga também


@pytest.mark.parametrize("name", CODECS)
def test_documented_codec_delays_match_the_measured_ones(name, tmp_path):
    # CODEC_DELAY_SAMPLES é removido antes de fatiar; se um build novo do codec
    # mudar o atraso, as janelas dos canais deixam de alinhar com as do canal limpo.
    audio = voiced_like(seconds=5.0)
    measured = delay_in_samples(audio, through(CODECS[name], tmp_path, audio))
    assert abs(measured - channel_sim.CODEC_DELAY_SAMPLES[name]) <= 3


def test_vonr_packet_loss_is_reproducible_and_changes_the_signal(tmp_path):
    audio = voiced_like()
    clean = through(channel_sim.to_vonr, tmp_path, audio)
    lossy_a = through(channel_sim.to_vonr, tmp_path, audio, fer=0.10, seed=3)
    lossy_b = through(channel_sim.to_vonr, tmp_path, audio, fer=0.10, seed=3)
    n = min(len(clean), len(lossy_a))
    assert np.array_equal(lossy_a, lossy_b)
    assert not np.allclose(clean[:n], lossy_a[:n])


def test_normalization_removes_level_differences_between_recordings(tmp_path):
    levels = []
    for i, gain in enumerate((0.05, 0.4)):
        src_path, dst_path = tmp_path / f"in{i}.wav", tmp_path / f"out{i}.wav"
        sf.write(src_path, gain * white_noise(1.5, seed=i) / 0.2, SR)
        channel_sim.normalize_clean(src_path, dst_path)
        out, sr = sf.read(dst_path, dtype="float32")
        assert sr == SR and len(out) == SR  # corta em exatamente 1 s
        levels.append(np.sqrt(np.mean(out ** 2)))
    assert 0.9 < levels[0] / levels[1] < 1.1  # dentro de ~1 dB, partindo de 18 dB de diferença
