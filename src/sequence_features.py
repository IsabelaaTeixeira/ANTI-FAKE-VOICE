"""Sequência de vetores do Wav2Vec2 (camada W2V_LAYER) por janela de 1 s, para
o classificador de sequência (src/sequence_classifier.py).

Os outros classificadores do projeto usam a MÉDIA no tempo dos 49 vetores que
o Wav2Vec2 produz para cada janela de 1 s (um a cada 20 ms). O classificador
de sequência lê os 49 vetores, para poder usar como eles variam ao longo do
segundo. Guardar 49 x 768 números por janela não caberia no disco para
centenas de milhares de janelas, então cada vetor é projetado em 128
dimensões por uma PCA ajustada SÓ com janelas do split de treino do ASVspoof
2019 (`fit_pca`). A rede é cortada na camada usada (as camadas acima não
mudam a saída dela), como no teste de latência.

Também devolve a média no tempo dos 768 valores — exatamente o atributo dos
classificadores de média —, para que as duas representações saiam da mesma
passada pela rede.
"""
from pathlib import Path

import numpy as np

from src.features import SR, W2V_LAYER, load_wav2vec2_model

PCA_PATH = Path("models/frame_pca.npz")
PCA_DIM = 128
N_FRAMES = 49  # quadros de 20 ms que o Wav2Vec2 produz para 1 s de áudio a 16 kHz


def load_layer_model():
    """Wav2Vec2 cortado na camada W2V_LAYER (a saída é a mesma da rede inteira)."""
    return load_wav2vec2_model(n_layers=W2V_LAYER)


def layer_frames(audios: list[np.ndarray], processor, model) -> np.ndarray:
    """(lote, 49, 768) em float32: os vetores da camada W2V_LAYER, quadro a quadro."""
    import torch

    inputs = processor(audios, sampling_rate=SR, return_tensors="pt", padding=True)
    with torch.inference_mode():
        return model(inputs.input_values).last_hidden_state.numpy()


def fit_pca(frames: np.ndarray, seed: int = 42) -> dict:
    """Ajusta a PCA em vetores de quadros (n, 768) e guarda média, componentes e
    a escala de cada componente (para a saída ter variância ~1 por dimensão)."""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=PCA_DIM, svd_solver="randomized", random_state=seed).fit(frames)
    params = {
        "mean": pca.mean_.astype(np.float32),
        "components": pca.components_.astype(np.float32),
        "scale": np.sqrt(pca.explained_variance_).astype(np.float32),
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32),
    }
    PCA_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(PCA_PATH, **params)
    return params


def load_pca() -> dict:
    if not PCA_PATH.exists():
        raise SystemExit(f"PCA não encontrada em {PCA_PATH}: rode antes `python -m src.build_dataset_asvspoof_full fit-pca`.")
    return dict(np.load(PCA_PATH))


def project(frames: np.ndarray, pca: dict) -> np.ndarray:
    """(lote, 49, 768) -> (lote, 49, 128) em float16."""
    z = (frames - pca["mean"]) @ pca["components"].T / pca["scale"]
    return z.astype(np.float16)


def encode_windows(audios: list[np.ndarray], processor, model, pca: dict, batch_size: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Janelas de 1 s -> (média no tempo (n, 768) float16, sequência projetada (n, 49, 128) float16)."""
    pooled, seqs = [], []
    for i in range(0, len(audios), batch_size):
        frames = layer_frames(audios[i:i + batch_size], processor, model)
        pooled.append(frames.mean(axis=1).astype(np.float16))
        seqs.append(project(frames, pca))
    if not pooled:
        return np.zeros((0, 768), np.float16), np.zeros((0, N_FRAMES, PCA_DIM), np.float16)
    return np.concatenate(pooled), np.concatenate(seqs)
