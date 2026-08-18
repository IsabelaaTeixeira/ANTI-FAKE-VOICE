"""Clona a voz da usuária via XTTS v2 (Coqui TTS), a partir de uma amostra
de referência, e salva o áudio clonado para testar contra o classificador.

IMPORTANTE: roda num ambiente virtual SEPARADO (.venv-tts/), porque o
Coqui TTS exige uma versão de `transformers` incompatível com a usada no
resto do projeto (Wav2Vec2). Não importa nada de `src/` — é só um script
solto, chamado com o Python de .venv-tts/.

Uso:
    .venv-tts/bin/python src/clone_voice.py
"""
from pathlib import Path

from TTS.api import TTS

REFERENCE_WAV = Path("data/my_voice_reference.wav")
OUTPUT_WAV = Path("data/cloned_voice.wav")

TEXT = (
    "Oi, aqui é um teste de clonagem de voz para o projeto de detecção de deepfake. "
    "Se você está ouvindo isso, significa que a clonagem funcionou."
)


def main() -> None:
    if not REFERENCE_WAV.exists():
        raise SystemExit(f"Referência não encontrada em {REFERENCE_WAV}.")

    print("Carregando o XTTS v2 (baixa o modelo na primeira vez, pode demorar)...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

    print("Gerando fala na sua voz clonada...")
    tts.tts_to_file(
        text=TEXT,
        speaker_wav=str(REFERENCE_WAV),
        language="pt",
        file_path=str(OUTPUT_WAV),
    )
    print(f"Áudio clonado salvo em {OUTPUT_WAV}")


if __name__ == "__main__":
    main()
