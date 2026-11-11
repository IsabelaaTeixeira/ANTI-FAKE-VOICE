"""Clona a voz da usuária via XTTS v2 (Coqui TTS), a partir de uma amostra
de referência, e salva o áudio clonado para testar contra o classificador.

Cada sessão de gravação da voz real vira uma voz clonada correspondente:
    data/my_voice_reference.wav    -> data/cloned_voice.wav
    data/my_voice_reference_2.wav  -> data/cloned_voice_2.wav
    ...
Sessões já clonadas são puladas. Com duas ou mais sessões, src/own_voice_ceiling.py
treina com as primeiras e testa na última (real e clonada).

IMPORTANTE: roda num ambiente virtual SEPARADO (.venv-tts/), porque o
Coqui TTS exige uma versão de `transformers` incompatível com a usada no
resto do projeto (Wav2Vec2). Não importa nada de `src/` — é só um script
solto, chamado com o Python de .venv-tts/.

Uso:
    .venv-tts/bin/python src/clone_voice.py
"""
from pathlib import Path

DATA_DIR = Path("data")
REFERENCE_PREFIX = "my_voice_reference"
OUTPUT_PREFIX = "cloned_voice"

TEXT = (
    "Oi, aqui é um teste de clonagem de voz para o projeto de detecção de deepfake. "
    "Se você está ouvindo isso, significa que a clonagem funcionou."
)


def pending_sessions() -> list[tuple[Path, Path]]:
    pairs = []
    for reference in sorted(DATA_DIR.glob(f"{REFERENCE_PREFIX}*.wav")):
        output = DATA_DIR / reference.name.replace(REFERENCE_PREFIX, OUTPUT_PREFIX, 1)
        if output.exists():
            print(f"{reference} já tem a voz clonada em {output}; pulando.")
        else:
            pairs.append((reference, output))
    return pairs


def main() -> None:
    if not list(DATA_DIR.glob(f"{REFERENCE_PREFIX}*.wav")):
        raise SystemExit(f"Nenhuma referência encontrada: grave {DATA_DIR / (REFERENCE_PREFIX + '.wav')} (ver README).")
    pairs = pending_sessions()
    if not pairs:
        return

    from TTS.api import TTS

    print("Carregando o XTTS v2 (baixa o modelo na primeira vez, pode demorar)...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    for reference, output in pairs:
        print(f"Gerando fala na sua voz clonada a partir de {reference}...")
        tts.tts_to_file(text=TEXT, speaker_wav=str(reference), language="pt", file_path=str(output))
        print(f"Áudio clonado salvo em {output}")


if __name__ == "__main__":
    main()
