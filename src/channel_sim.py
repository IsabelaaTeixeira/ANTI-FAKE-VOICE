"""Simuladores de canal de telecomunicação (GSM, VoLTE, VoNR/EVS, VoNR com
perda de pacotes).

Usa um build próprio do FFmpeg com os encoders libopencore_amrnb e
libvo_amrwbenc (a build padrão do apt no Ubuntu não os inclui), compilado em
tools/ffmpeg-amr/, e o codec de referência EVS (3GPP TS 26.443) compilado em
tools/evs-src/ — ver README, seção "Setup".

Os codecs aqui processam a gravação INTEIRA, sem cortar em 1s e sem
normalizar: o fatiamento em janelas e a normalização de loudness acontecem
depois, em build_dataset.slice_to_windows. Assim o codec roda contínuo, como
numa ligação de verdade — se cada janela de 1s fosse codificada sozinha, o
codec começaria "frio" (sem histórico do sinal) a cada segundo, um artefato
que não existe numa chamada real.
"""
import os
import subprocess
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
FFMPEG = _PROJECT_ROOT / "tools" / "ffmpeg-amr" / "bin" / "ffmpeg"
EVS_ENCODER = _PROJECT_ROOT / "tools" / "evs-src" / "EVS_cod"
EVS_DECODER = _PROJECT_ROOT / "tools" / "evs-src" / "EVS_dec"
EVS_BITRATE = "13200"  # 13.2 kb/s: taxa padrão usada em VoNR/5G

# Formato G.192 do EVS (ver tools/evs-src/readme.txt): cada quadro de 20ms começa
# com uma palavra de sincronia; 0x6B20 diz ao decodificador que o quadro se perdeu,
# e ele aplica a ocultação de perda (PLC) do próprio EVS para "inventar" o trecho.
G192_SYNC_GOOD = 0x6B21
G192_SYNC_BAD = 0x6B20

# Perda de pacotes: taxas de quadros perdidos (FER) sorteadas por gravação, na faixa
# usada nos testes de robustez do EVS (3–10%), em rajadas (modelo de Gilbert-Elliott:
# quando um pacote se perde, o seguinte tem mais chance de se perder também).
FER_CHOICES = (0.03, 0.06, 0.10)
MEAN_BURST_FRAMES = 2.0

# Atraso algorítmico de cada codec (em amostras a 16kHz), medido por correlação
# cruzada entre a entrada e a saída decodificada de 60s de fala: build_dataset
# descarta esse início antes de fatiar, para que a janela i de um canal com codec
# contenha o mesmo trecho de fala da janela i do canal limpo.
CODEC_DELAY_SAMPLES = {"gsm": 79, "volte": 95, "vonr": 0, "vonr_loss": 0}


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


def to_gsm(src: Path, dst: Path) -> None:
    """AMR-NB 12.2 kb/s, banda ~300-3400 Hz (mesma config do artigo VoLTE/GSM)."""
    tmp = dst.with_name(dst.stem + "_gsm.amr")
    _run([
        str(FFMPEG), "-y", "-i", str(src),
        "-ar", "8000", "-ac", "1",
        "-c:a", "libopencore_amrnb", "-b:a", "12.2k",
        "-f", "amr",
        str(tmp),
    ])
    _decode_to_wav(tmp, dst)


def to_volte(src: Path, dst: Path) -> None:
    """AMR-WB 12.65 kb/s, banda ~50-7000 Hz."""
    tmp = dst.with_name(dst.stem + "_volte.amr")
    _run([
        str(FFMPEG), "-y", "-i", str(src),
        "-ar", "16000", "-ac", "1",
        "-c:a", "libvo_amrwbenc", "-b:a", "12.65k",
        "-f", "amr",
        str(tmp),
    ])
    _decode_to_wav(tmp, dst)


def to_vonr(src: Path, dst: Path, fer: float = 0.0, seed: int | None = None) -> None:
    """VoNR aproximado pelo codec de referência EVS (3GPP TS 26.443), 13.2 kb/s.

    O EVS_cod/EVS_dec trabalham com PCM cru (sem cabeçalho), então convertemos
    WAV -> PCM -> EVS -> PCM -> WAV. Com fer > 0, marca quadros como perdidos
    entre o codificador e o decodificador (ver drop_g192_frames).
    """
    raw_in = dst.with_name(dst.stem + "_vonr_in.16k")
    stream = dst.with_name(dst.stem + "_vonr.192")
    raw_out = dst.with_name(dst.stem + "_vonr_out.16k")

    _run([str(FFMPEG), "-y", "-i", str(src), "-ar", "16000", "-ac", "1", "-f", "s16le", str(raw_in)])
    _run([str(EVS_ENCODER), EVS_BITRATE, "16", str(raw_in), str(stream)])
    if fer > 0:
        drop_g192_frames(stream, fer, np.random.default_rng(seed))
    _run([str(EVS_DECODER), "16", str(stream), str(raw_out)])
    _run([
        str(FFMPEG), "-y", "-f", "s16le", "-ar", "16000", "-ac", "1", "-i", str(raw_out),
        "-ar", "16000", "-ac", "1",
        str(dst),
    ])
    for tmp in (raw_in, stream, raw_out):
        tmp.unlink(missing_ok=True)


def to_vonr_loss(src: Path, dst: Path, seed: int | None = None) -> None:
    """VoNR (EVS 13.2 kb/s) com perda de pacotes em rajada; a taxa de perda é
    sorteada entre FER_CHOICES a cada gravação."""
    rng = np.random.default_rng(seed)
    fer = float(rng.choice(FER_CHOICES))
    to_vonr(src, dst, fer=fer, seed=int(rng.integers(0, 2**32)))


def drop_g192_frames(stream: Path, fer: float, rng: np.random.Generator) -> float:
    """Marca quadros do bitstream G.192 como perdidos (sync 0x6B20), seguindo um
    modelo de Gilbert-Elliott com taxa média de perda `fer` e rajadas de
    MEAN_BURST_FRAMES quadros em média. Retorna a taxa de perda realizada."""
    words = np.fromfile(stream, dtype="<u2")
    frame_starts = []
    pos = 0
    while pos + 1 < len(words):
        frame_starts.append(pos)
        pos += 2 + int(words[pos + 1])

    p_bad_to_good = 1.0 / MEAN_BURST_FRAMES
    p_good_to_bad = fer * p_bad_to_good / (1.0 - fer)
    lost = False
    n_lost = 0
    for start in frame_starts:
        lost = rng.random() < ((1.0 - p_bad_to_good) if lost else p_good_to_bad)
        if lost:
            words[start] = G192_SYNC_BAD
            n_lost += 1
    words.tofile(stream)
    return n_lost / max(len(frame_starts), 1)


def _decode_to_wav(encoded: Path, dst: Path) -> None:
    """Decodifica de volta para WAV 16kHz mono (sem normalizar: isso é feito por janela)."""
    _run([
        str(FFMPEG), "-y", "-i", str(encoded),
        "-ar", "16000", "-ac", "1",
        str(dst),
    ])
    encoded.unlink(missing_ok=True)


def normalize_clean(src: Path, dst: Path) -> None:
    """Corta em 1s e normaliza o loudness a -16 LUFS (padrão de fala). Aplicada a
    TODA janela, de qualquer canal, depois do fatiamento — assim nenhum canal
    (nem o clean) difere dos outros em volume, o que corrige a assimetria de
    volume entre REAL e FAKE encontrada na auditoria de silêncio (fase T).

    Escreve num arquivo temporário e só renomeia para `dst` no fim: se o
    processo for interrompido no meio, nunca sobra um `dst` incompleto que
    uma execução retomada confundiria com uma janela pronta."""
    partial = dst.with_name(dst.name + ".partial")
    _run([
        str(FFMPEG), "-y", "-i", str(src),
        "-ar", "16000", "-ac", "1", "-t", "1.0",
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-f", "wav", str(partial),
    ])
    os.replace(partial, dst)


def check_amr_encoders_available() -> bool:
    """Confere se o FFmpeg compilado tem os encoders AMR necessários."""
    if not FFMPEG.exists():
        return False
    out = subprocess.run([str(FFMPEG), "-encoders"], capture_output=True, text=True).stdout
    return "libopencore_amrnb" in out and "libvo_amrwbenc" in out


def check_evs_available() -> bool:
    """Confere se os binários do codec de referência EVS foram compilados."""
    return EVS_ENCODER.exists() and EVS_DECODER.exists()


if __name__ == "__main__":
    if not FFMPEG.exists():
        raise SystemExit(
            f"FFmpeg com AMR não encontrado em {FFMPEG}.\n"
            "Compile-o primeiro (ver README, seção 'Setup')."
        )
    if not check_amr_encoders_available():
        raise SystemExit("FFmpeg encontrado, mas sem os encoders AMR esperados.")
    print(f"FFmpeg com encoders AMR: OK ({FFMPEG})")

    if not check_evs_available():
        raise SystemExit(f"Binários do EVS não encontrados em {EVS_ENCODER.parent}.")
    print(f"Codec EVS (VoNR): OK ({EVS_ENCODER.parent})")
