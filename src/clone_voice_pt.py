"""Gera as vozes clonadas do conjunto de teste em português (ver
src/build_dataset_pt.py) com dois sintetizadores de clonagem de voz:
XTTS v2 (Coqui, 2023) e YourTTS (2022, treinado com português do Brasil).

Para cada linha de data/raw/PT/clone_jobs.csv, clona a voz do locutor a
partir da gravação de referência dele e fala o texto indicado. Pula o que
já existe, então pode ser interrompido e retomado.

IMPORTANTE: como src/clone_voice.py, roda no ambiente virtual SEPARADO
(.venv-tts/) e não importa nada de `src/`. Ao usar, você concorda com as
licenças dos modelos: CPML da Coqui (XTTS v2) e CC BY-NC 4.0 (YourTTS) —
uso de pesquisa, não comercial.

Uso:
    COQUI_TOS_AGREED=1 .venv-tts/bin/python src/clone_voice_pt.py
"""
import csv
from pathlib import Path

from tqdm import tqdm
from TTS.api import TTS

JOBS_PATH = Path("data/raw/PT/clone_jobs.csv")
FAKE_DIR = Path("data/raw/PT/fake")
GENERATORS = {
    "xtts_v2": ("tts_models/multilingual/multi-dataset/xtts_v2", "pt"),
    "your_tts": ("tts_models/multilingual/multi-dataset/your_tts", "pt-br"),
}


def main() -> None:
    with open(JOBS_PATH, newline="") as f:
        jobs = list(csv.DictReader(f))
    for generator, (model_name, language) in GENERATORS.items():
        out_dir = FAKE_DIR / generator
        out_dir.mkdir(parents=True, exist_ok=True)
        pending = [j for j in jobs if not (out_dir / f"{generator}_{j['job_id']}.wav").exists()]
        if not pending:
            continue
        print(f"Carregando {generator} ({model_name})...")
        tts = TTS(model_name)
        for job in tqdm(pending, desc=generator):
            out_path = out_dir / f"{generator}_{job['job_id']}.wav"
            tmp_path = out_path.with_suffix(".tmp.wav")
            tts.tts_to_file(text=job["text"], speaker_wav=job["reference_path"], language=language, file_path=str(tmp_path))
            tmp_path.rename(out_path)


if __name__ == "__main__":
    main()
