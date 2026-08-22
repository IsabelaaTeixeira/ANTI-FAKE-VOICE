"""Simuladores de canal de telecomunicação (GSM, VoLTE, VoNR/EVS).

Usa um build próprio do FFmpeg com os encoders libopencore_amrnb e
libvo_amrwbenc (a build padrão do apt no Ubuntu não os inclui), compilado em
tools/ffmpeg-amr/, e o codec de referência EVS (3GPP TS 26.443) compilado em
tools/evs-src/ — ver README, seção "Setup".
"""
import subprocess
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
FFMPEG = _PROJECT_ROOT / "tools" / "ffmpeg-amr" / "bin" / "ffmpeg"
EVS_ENCODER = _PROJECT_ROOT / "tools" / "evs-src" / "EVS_cod"
EVS_DECODER = _PROJECT_ROOT / "tools" / "evs-src" / "EVS_dec"
EVS_BITRATE = "13200"  # 13.2 kb/s: taxa padrão usada em VoNR/5G


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


def to_vonr(src: Path, dst: Path) -> None:
    """VoNR aproximado pelo codec de referência EVS (3GPP TS 26.443), 13.2 kb/s.

    O EVS_cod/EVS_dec trabalham com PCM cru (sem cabeçalho), então convertemos
    WAV -> PCM -> EVS -> PCM -> WAV, normalizando no final como nos outros canais.
    """
    raw_in = dst.with_name(dst.stem + "_vonr_in.16k")
    stream = dst.with_name(dst.stem + "_vonr.192")
    raw_out = dst.with_name(dst.stem + "_vonr_out.16k")

    _run([str(FFMPEG), "-y", "-i", str(src), "-ar", "16000", "-ac", "1", "-f", "s16le", str(raw_in)])
    _run([str(EVS_ENCODER), EVS_BITRATE, "16", str(raw_in), str(stream)])
    _run([str(EVS_DECODER), "16", str(stream), str(raw_out)])
    _run([
        str(FFMPEG), "-y", "-f", "s16le", "-ar", "16000", "-ac", "1", "-i", str(raw_out),
        "-ar", "16000", "-ac", "1", "-t", "1.0",
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        str(dst),
    ])
    for tmp in (raw_in, stream, raw_out):
        tmp.unlink(missing_ok=True)


def _decode_to_wav(encoded: Path, dst: Path) -> None:
    """Decodifica de volta para WAV 16kHz mono, normalizado a -16 LUFS (padrão de fala)."""
    _run([
        str(FFMPEG), "-y", "-i", str(encoded),
        "-ar", "16000", "-ac", "1", "-t", "1.0",
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        str(dst),
    ])
    encoded.unlink(missing_ok=True)


def normalize_clean(src: Path, dst: Path) -> None:
    """Aplica a MESMA normalização de loudness usada em gsm/volte/vonr (-16 LUFS)
    ao canal 'clean', sem nenhum codec — corrige a assimetria de volume entre
    REAL e FAKE encontrada na auditoria de silêncio (fase T do relatório):
    o canal clean nunca passava por normalização, só os canais com codec."""
    _run([
        str(FFMPEG), "-y", "-i", str(src),
        "-ar", "16000", "-ac", "1", "-t", "1.0",
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        str(dst),
    ])


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
