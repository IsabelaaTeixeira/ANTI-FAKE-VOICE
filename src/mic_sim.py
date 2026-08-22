"""Simula gravação por um microfone/ambiente desconhecido (reverb + ruído de
fundo + coloração de frequência de microfone barato), em vez de um codec de
telecom — ataca o "channel effect" de captura documentado por Zhang et al.
(fase S/U) e confirmado na prática pelo teste ao vivo da fase W, onde a voz
real gravada num notebook era confundida com voz clonada por um modelo só
treinado com augmentation de codec (GSM/VoLTE/VoNR).

A reverberação é sintética (ruído branco com envelope de decaimento
exponencial, parametrizado por RT60) em vez de uma resposta de impulso real
gravada — mais simples, sem depender de baixar um banco de RIRs, e ajustável
por sorteio (sala/mic diferentes a cada chamada). Termina passando pelo mesmo
`channel_sim.normalize_clean` usado no canal limpo, para manter a mesma
normalização de loudness (-16 LUFS) dos demais canais.

Dois perfis (fase AA/BB/CC/DD): o teste ao vivo mostrou que o mic embutido do
notebook e um headset close-talk se comportam de forma bem diferente. A
primeira versão do perfil "close_talk" (fase BB) chutava um reforço de graves
por "efeito de proximidade" — testado ao vivo (fase CC), não funcionou, o
resultado ficou igual ao de antes de existir o perfil.

Fase DD: em vez de chutar parâmetros, gravamos a mesma fala da usuária pelos
dois microfones (data/calib_notebook.wav, data/calib_headset.wav) e medimos a
diferença real de espectro (Welch, só nos trechos de fala ativa, ignorando o
ruído). O resultado contrariou a hipótese original: o notebook tinha muito
mais energia grave (<100Hz), mas por RUÍDO de ambiente/ventoinha, não por
timbre de voz — descontando isso, o headset tem proporcionalmente MAIS
"presença" em 300-1200Hz, não mais grave. `EQ_CLOSE_TALK_GAIN_DB` é essa
curva medida (com fator de confiança 0,6x, por vir de uma única gravação de
15s), aplicada como filtro FIR (`scipy.signal.firwin2`) em vez do shelf de
graves chutado da fase BB.

`to_mic()` sorteia entre "room" (o perfil original, mic embutido do notebook)
e "close_talk" (reverb quase nula, mais silencioso, EQ medido) quando chamado
com profile="random" (o padrão usado pelo pipeline de treino).
"""
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from src import channel_sim

SR = 16000

# Perfis sorteados a cada chamada, para simular salas/microfones variados.
# "room": ajustado na fase Y após o 1º teste ficar exagerado (RT60 até 0,6s +
#   SNR até 12dB deixou a condição bem mais dura que uma sala normal com
#   notebook — o teste real da fase W era um quarto comum, não um salão vazio
#   e ruidoso). Representa o mic embutido do notebook.
# "close_talk": reverb quase nula (mic muito perto da fonte) e SNR mais alto
#   (pouco ruído de ambiente chega tão perto) — essas duas partes já eram
#   plausíveis fisicamente e não foram contestadas pela medição da fase DD.
#   O que mudou na fase DD foi a coloração de frequência: em vez do shelf de
#   graves chutado (fase BB, não validado ao vivo), usa o EQ medido em
#   EQ_CLOSE_TALK_GAIN_DB.
PROFILES = {
    "room": dict(
        rt60_range=(0.08, 0.35),
        snr_db_range=(18, 35),
        highpass_hz_range=(50, 150),
        lowpass_hz_range=(7000, 8000),
        dry_wet_mix=0.5,
    ),
    "close_talk": dict(
        rt60_range=(0.005, 0.04),
        snr_db_range=(28, 42),
        highpass_hz_range=(20, 50),
        lowpass_hz_range=(7500, 8000),
        dry_wet_mix=0.9,
    ),
}

# Curva de EQ medida na fase DD (Welch, frames de fala ativa, headset/notebook
# em dB, escalado por 0,6x de confiança e limitado a +-4dB por vir de uma
# única amostra de 15s). Pontos de âncora em Hz -> ganho em dB.
_EQ_FREQ_HZ = np.array([0, 80, 125, 225, 450, 900, 1850, 3250, 5000, 7000, 8000])
_EQ_GAIN_DB_RAW = np.array([-2.0, -2.0, 1.77, 2.03, 2.90, 5.44, 3.33, 1.45, 0.90, -1.6, -1.6])
EQ_CONFIDENCE = 0.6
EQ_CLOSE_TALK_GAIN_DB = np.clip(_EQ_GAIN_DB_RAW * EQ_CONFIDENCE, -4, 4)
_EQ_TAPS = signal.firwin2(513, _EQ_FREQ_HZ / (SR / 2), 10 ** (EQ_CLOSE_TALK_GAIN_DB / 20))


def _synthetic_rir(rt60: float, rng: np.random.Generator) -> np.ndarray:
    length = max(int(rt60 * SR), 8)
    t = np.arange(length) / SR
    decay = np.exp(-6.9 * t / rt60)  # -60dB em t=rt60 (6.9 ~= ln(1000))
    rir = rng.standard_normal(length) * decay
    peak = np.max(np.abs(rir))
    return (rir / peak if peak > 0 else rir).astype(np.float32)


def to_mic(src: Path, dst: Path, seed: int | None = None, profile: str = "random") -> None:
    rng = np.random.default_rng(seed)
    if profile == "random":
        profile = "room" if rng.integers(0, 2) == 0 else "close_talk"
    params = PROFILES[profile]

    y, sr_in = sf.read(src, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr_in != SR:
        y = signal.resample(y, int(len(y) * SR / sr_in)).astype(np.float32)

    rir = _synthetic_rir(rng.uniform(*params["rt60_range"]), rng)
    wet = signal.fftconvolve(y, rir, mode="full")[: len(y)]
    dry_wet = params["dry_wet_mix"]
    mix = (1 - dry_wet) * wet + dry_wet * y

    hp = rng.uniform(*params["highpass_hz_range"])
    lp = rng.uniform(*params["lowpass_hz_range"])
    sos = signal.butter(2, [hp, lp], btype="bandpass", fs=SR, output="sos")
    colored = signal.sosfilt(sos, mix)

    if profile == "close_talk":
        colored = signal.fftconvolve(colored, _EQ_TAPS, mode="same")

    snr_db = rng.uniform(*params["snr_db_range"])
    sig_power = np.mean(colored ** 2) + 1e-12
    noise_power = sig_power / (10 ** (snr_db / 10))
    noisy = colored + rng.standard_normal(len(colored)) * np.sqrt(noise_power)

    peak = np.max(np.abs(noisy))
    if peak > 0:
        noisy = noisy / peak * 0.9

    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_path = Path(tmp_dir) / "mic_raw.wav"
        sf.write(raw_path, noisy.astype(np.float32), SR)
        channel_sim.normalize_clean(raw_path, dst)
