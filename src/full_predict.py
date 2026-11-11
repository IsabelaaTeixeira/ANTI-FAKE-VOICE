"""Aplica os modelos treinados no split de treino do ASVspoof 2019 aos conjuntos de
teste, numa passada só pela rede (Wav2Vec2 cortado na camada W2V_LAYER), sem
guardar a sequência de vetores dos testes em disco — só as predições.

Modelos (todos sobre a camada W2V_LAYER, nos três regimes de treino):
  - lr_sample_<regime>: média + Regressão Logística, treinada na amostra do
    split de treino (2.683 gravações; models/wav2vec2_<regime>_asvspoof.joblib);
  - lr_full_<regime>: o mesmo classificador, no split de treino completo
    (25.380 gravações; src/sequence_classifier.py);
  - seq_full_<regime>: o classificador de sequência, no split completo.

Fontes de teste:
  - asvspoof: dev e eval da amostra do ASVspoof 2019, nos 6 canais simulados
    (janelas já em disco, data/manifest_asvspoof.csv);
  - perwindow: as mesmas gravações do eval, mas com o codec aplicado a CADA
    JANELA DE 1 s ISOLADA (GSM, VoLTE, VoNR), em vez da gravação inteira — o
    controle que mede o efeito da forma de simular o canal; tudo o mais
    (corte de silêncio, nível de entrada, compensação de atraso,
    normalização) é igual ao da simulação contínua;
  - pt: o conjunto em português, nos 6 canais (data/manifest_pt.csv);
  - asvspoof2021: o ASVspoof 2021 LA eval inteiro, lido em sequência do
    .tar.gz (ver src/build_dataset_asvspoof2021.py).

As predições saem em blocos (data/predictions_full/<fonte>/part_*.csv); uma
execução interrompida retoma do último bloco completo.

Uso:
    python -m src.full_predict --source asvspoof|perwindow|pt|asvspoof2021
"""
import argparse
import tempfile
from multiprocessing import Pool
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.build_dataset import apply_codec_to_recording, _file_seed, load_trimmed
from src.build_dataset_asvspoof import RAW_DIR, SPLIT_DIRS
from src.build_dataset_asvspoof_full import normalize
from src.features import SR
from src.sequence_classifier import load_seqnet, lr_model_path, predict_seqnet
from src.train import REGIMES, final_model_path

OUT_DIR = Path("data/predictions_full")
SOURCES = ("asvspoof", "perwindow", "pt", "asvspoof2021")
BLOCK_WINDOWS = 4096
BLOCK_RECORDINGS = 600
N_WORKERS = 6
TORCH_THREADS = 4
PERWINDOW_CHANNELS = ("gsm", "volte", "vonr")
WINDOW = SR


class Scorer:
    def __init__(self):
        self.lr = {}
        for regime in REGIMES:
            self.lr[f"lr_sample_{regime}"] = joblib.load(final_model_path("wav2vec2", regime, "asvspoof"))
            self.lr[f"lr_full_{regime}"] = joblib.load(lr_model_path(regime))
        self.seq = {f"seq_full_{regime}": load_seqnet(regime) for regime in REGIMES}

    def score(self, pooled: np.ndarray, seq: np.ndarray) -> dict[str, np.ndarray]:
        pooled32 = pooled.astype(np.float32)
        out = {name: clf.predict_proba(pooled32)[:, 1] for name, clf in self.lr.items()}
        out.update({name: predict_seqnet(model, seq, pooled32) for name, model in self.seq.items()})
        return out


def write_block(out_dir: Path, index: int, meta: pd.DataFrame, scores: dict[str, np.ndarray]) -> None:
    out = meta.reset_index(drop=True).assign(**scores)
    tmp = out_dir / f"part_{index:05d}.csv.tmp"
    out.to_csv(tmp, index=False)
    tmp.rename(out_dir / f"part_{index:05d}.csv")


def prepare_perwindow(job: tuple[str, dict]) -> tuple[dict, dict[str, list[np.ndarray]]]:
    """Gravação do eval -> janelas com o codec aplicado a cada janela isolada."""
    flac_path, info = job
    audio_path = Path(flac_path)
    audio = load_trimmed(audio_path)
    n = len(audio) // WINDOW
    windows = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for channel in PERWINDOW_CHANNELS:
            windows[channel] = [
                normalize(apply_codec_to_recording(audio[i * WINDOW:(i + 1) * WINDOW], channel, _file_seed(audio_path), tmp_dir), tmp_dir)
                for i in range(n)
            ]
    return info, windows


def iter_file_blocks(manifest: pd.DataFrame):
    for start in range(0, len(manifest), BLOCK_WINDOWS):
        rows = manifest.iloc[start:start + BLOCK_WINDOWS]
        audios = [librosa.load(p, sr=SR)[0] for p in rows["filepath"]]
        yield rows.drop(columns=["filepath"]), audios


def run(source: str) -> None:
    import torch

    from src.sequence_features import encode_windows, load_layer_model, load_pca

    torch.set_num_threads(TORCH_THREADS)
    out_dir = OUT_DIR / source
    out_dir.mkdir(parents=True, exist_ok=True)
    done = len(list(out_dir.glob("part_*.csv")))

    if source in ("asvspoof", "pt"):
        manifest = pd.read_csv(f"data/manifest_{source}.csv")
        if source == "asvspoof":
            manifest = manifest[manifest["split"].isin(["dev", "eval"])].reset_index(drop=True)
        n_blocks = int(np.ceil(len(manifest) / BLOCK_WINDOWS))
        print(f"{source}: {len(manifest)} janelas em {n_blocks} blocos; {done} já prontos.")
        processor, model = load_layer_model()
        pca, scorer = load_pca(), Scorer()
        blocks = iter_file_blocks(manifest.iloc[done * BLOCK_WINDOWS:])
        for b, (meta, audios) in enumerate(tqdm(blocks, initial=done, total=n_blocks, desc=source), start=done):
            pooled, seq = encode_windows(audios, processor, model, pca)
            write_block(out_dir, b, meta, scorer.score(pooled, seq))
        return

    if source == "perwindow":
        manifest = pd.read_csv("data/manifest_asvspoof.csv")
        recs = manifest[(manifest["split"] == "eval") & (manifest["channel"] == "clean")].drop_duplicates("source_file")
        flac_dir = RAW_DIR / SPLIT_DIRS["eval"] / "flac"
        jobs = [(str(flac_dir / r.source_file), {"speakers": r.speakers, "label": r.label, "source_file": r.source_file,
                                                  "attack_id": r.attack_id, "split": "eval"}) for r in recs.itertuples()]
        blocks = [jobs[i:i + BLOCK_RECORDINGS] for i in range(0, len(jobs), BLOCK_RECORDINGS)]
        print(f"perwindow: {len(jobs)} gravações do eval em {len(blocks)} blocos; {done} já prontos.")
        if done >= len(blocks):
            return
        with Pool(N_WORKERS) as pool:
            processor, model = load_layer_model()
            pca, scorer = load_pca(), Scorer()
            pending = pool.map_async(prepare_perwindow, blocks[done], chunksize=4)
            for b in tqdm(range(done, len(blocks)), initial=done, total=len(blocks), desc="perwindow"):
                prepared = pending.get()
                if b + 1 < len(blocks):
                    pending = pool.map_async(prepare_perwindow, blocks[b + 1], chunksize=4)
                rows, audios = [], []
                for info, windows in prepared:
                    for channel, ws in windows.items():
                        for idx, w in enumerate(ws):
                            rows.append({**info, "channel": channel, "window_index": idx, "simulation": "per_window"})
                            audios.append(w)
                pooled, seq = encode_windows(audios, processor, model, pca)
                write_block(out_dir, b, pd.DataFrame(rows), scorer.score(pooled, seq))
        return

    # asvspoof2021: mesmo preparo e mesma leitura em sequência de src/build_dataset_asvspoof2021.py
    from src.build_dataset_asvspoof2021 import CHUNK_UTTERANCES, TAR_PATH, iter_eval_chunks, load_metadata, prepare_utterance

    if not TAR_PATH.exists():
        raise SystemExit(f"Pacote do ASVspoof 2021 não encontrado em {TAR_PATH} (ver README).")
    meta = load_metadata()
    n_blocks = int(np.ceil(len(meta) / CHUNK_UTTERANCES))
    print(f"asvspoof2021: {len(meta)} gravações em {n_blocks} blocos; {done} já prontos.")
    with Pool(N_WORKERS) as pool:
        processor, model = load_layer_model()
        pca, scorer = load_pca(), Scorer()
        chunks = iter_eval_chunks(set(meta.index), done * CHUNK_UTTERANCES)
        for b, chunk in enumerate(tqdm(chunks, initial=done, total=n_blocks, desc="asvspoof2021"), start=done):
            prepared = pool.map(prepare_utterance, chunk, chunksize=16)
            rows, audios = [], []
            for trial, windows in prepared:
                info = meta.loc[trial]
                for idx, w in enumerate(windows):
                    rows.append({"speakers": info["speaker"], "label": "real" if info["key"] == "bonafide" else "fake",
                                 "channel": info["codec"], "transmission": info["transmission"],
                                 "attack_id": info["attack_id"], "source_file": trial, "window_index": idx})
                    audios.append(w)
            pooled, seq = encode_windows(audios, processor, model, pca)
            write_block(out_dir, b, pd.DataFrame(rows), scorer.score(pooled, seq))


def load_predictions(source: str) -> pd.DataFrame:
    parts = sorted((OUT_DIR / source).glob("part_*.csv"))
    return pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=SOURCES, required=True)
    run(parser.parse_args().source)


if __name__ == "__main__":
    main()
