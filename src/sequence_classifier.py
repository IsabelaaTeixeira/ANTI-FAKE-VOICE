"""Treina, no split de treino COMPLETO do ASVspoof 2019 (src/build_dataset_asvspoof_full.py),
os dois classificadores sobre o Wav2Vec2 (camada W2V_LAYER), nos três regimes
de treino do projeto (limpo, augmentado e augmentado pareado):

  - média + Regressão Logística: o mesmo classificador do resto do projeto
    (src/train.make_classifier), agora com 10 vezes mais gravações;
  - classificador de sequência: uma rede pequena que lê os 49 vetores da
    janela (um a cada 20 ms, projetados em 128 dimensões pela PCA de
    src/sequence_features.py) com duas camadas convolucionais no tempo e um
    pooling de estatísticas com atenção (média e desvio-padrão ponderados por
    pesos que a rede aprende), mais a média completa dos 768 valores — a mesma
    informação da Regressão Logística, de modo que a diferença entre os dois
    mede o que a sequência acrescenta.

A rede é treinada com Adam e perda logística com peso por classe (as vozes
reais são ~10% do treino); a época final é escolhida pelo menor EER no dev da
amostra do ASVspoof 2019 (src/build_dataset_asvspoof_full.py dev), nos canais
do próprio regime — o limpo só vê o canal limpo também na escolha da época.
Nenhum dado de teste participa.

Uso:
    python -m src.sequence_classifier
"""
import copy
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.build_dataset_asvspoof_full import DEV_DIR, FULL_DIR
from src.evaluate import eer
from src.sequence_features import PCA_DIM
from src.train import RANDOM_SEED, REGIMES, TRAIN_CHANNELS, label_vector, make_classifier, regime_mask

MODELS_DIR = Path("models")
LOG_OUT = Path("results/full_train_log.csv")
BATCH = 256
MAX_EPOCHS = 30
PATIENCE = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TORCH_THREADS = 4


def lr_model_path(regime: str) -> Path:
    return MODELS_DIR / f"wav2vec2_{regime}_asvspoof_full.joblib"


def seq_model_path(regime: str) -> Path:
    return MODELS_DIR / f"seqnet_{regime}_asvspoof_full.pt"


class FeatureStore:
    """Blocos part_*.csv / .pooled.npy / .seq.npy de uma pasta, lidos por memmap
    (o treino completo não cabe inteiro na memória)."""

    def __init__(self, directory: Path):
        parts = sorted(directory.glob("part_*.csv"))
        if not parts:
            raise SystemExit(f"Nenhum bloco em {directory}: rode antes src.build_dataset_asvspoof_full.")
        self.meta = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        self.pooled = [np.load(str(p).replace(".csv", ".pooled.npy"), mmap_mode="r") for p in parts]
        self.seq = [np.load(str(p).replace(".csv", ".seq.npy"), mmap_mode="r") for p in parts]
        self.offsets = np.concatenate([[0], np.cumsum([len(a) for a in self.pooled])])

    def _gather(self, arrays: list, idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx)
        order = np.argsort(idx, kind="stable")
        sorted_idx = idx[order]
        part = np.searchsorted(self.offsets, sorted_idx, side="right") - 1
        gathered = np.empty((len(idx),) + arrays[0].shape[1:], dtype=np.float32)
        for p in np.unique(part):
            m = part == p
            gathered[m] = arrays[p][sorted_idx[m] - self.offsets[p]]
        out = np.empty_like(gathered)
        out[order] = gathered
        return out

    def pooled_rows(self, idx: np.ndarray) -> np.ndarray:
        return self._gather(self.pooled, idx)

    def seq_rows(self, idx: np.ndarray) -> np.ndarray:
        return self._gather(self.seq, idx)


def build_seqnet(pooled_mean: np.ndarray, pooled_std: np.ndarray):
    import torch
    from torch import nn

    class SeqNet(nn.Module):
        def __init__(self, d: int = 128):
            super().__init__()
            self.register_buffer("pooled_mean", torch.as_tensor(pooled_mean, dtype=torch.float32))
            self.register_buffer("pooled_std", torch.as_tensor(pooled_std, dtype=torch.float32))
            self.conv = nn.Sequential(
                nn.Conv1d(PCA_DIM, d, 3, padding=1), nn.BatchNorm1d(d), nn.ReLU(),
                nn.Conv1d(d, d, 3, padding=1), nn.BatchNorm1d(d), nn.ReLU(),
            )
            self.attention = nn.Sequential(nn.Conv1d(d, 64, 1), nn.Tanh(), nn.Conv1d(64, 1, 1))
            self.mean_branch = nn.Sequential(nn.Linear(len(pooled_mean), d), nn.ReLU())
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(3 * d, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

        def forward(self, seq, pooled):
            h = self.conv(seq.transpose(1, 2))                 # (lote, d, 49)
            w = torch.softmax(self.attention(h), dim=2)        # peso de cada quadro
            mu = (h * w).sum(dim=2)
            sd = ((h - mu.unsqueeze(2)).pow(2) * w).sum(dim=2).clamp_min(1e-6).sqrt()
            m = self.mean_branch((pooled - self.pooled_mean) / self.pooled_std)
            return self.head(torch.cat([mu, sd, m], dim=1)).squeeze(1)

    return SeqNet()


def predict_seqnet(model, seq: np.ndarray, pooled: np.ndarray, batch: int = 2048) -> np.ndarray:
    import torch

    model.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(seq), batch):
            logits = model(torch.as_tensor(seq[i:i + batch], dtype=torch.float32),
                           torch.as_tensor(pooled[i:i + batch], dtype=torch.float32))
            out.append(torch.sigmoid(logits).numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def load_seqnet(regime: str):
    import torch

    ckpt = torch.load(seq_model_path(regime), weights_only=False)
    model = build_seqnet(ckpt["pooled_mean"], ckpt["pooled_std"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def regime_indices(store: FeatureStore, regime: str) -> np.ndarray:
    channels = store.meta["channel"].values
    return np.flatnonzero(regime_mask(regime, channels, np.ones(len(channels), dtype=bool)))


def dev_indices(dev: FeatureStore, regime: str) -> np.ndarray:
    channels = ["clean"] if regime == "clean" else list(TRAIN_CHANNELS)
    return np.flatnonzero(dev.meta["channel"].isin(channels).values)


def train_lr(store: FeatureStore, dev: FeatureStore, regime: str) -> dict:
    idx = regime_indices(store, regime)
    y = label_vector(store.meta)[idx]
    t0 = time.perf_counter()
    clf = make_classifier("wav2vec2").fit(store.pooled_rows(idx), y)
    joblib.dump(clf, lr_model_path(regime))
    d = dev_indices(dev, regime)
    dev_eer = eer(label_vector(dev.meta)[d], clf.predict_proba(dev.pooled_rows(d))[:, 1])
    print(f"Regressão Logística · {regime}: {len(idx)} janelas, EER no dev {dev_eer:.3f} ({time.perf_counter() - t0:.0f} s)")
    return {"model": "lr_full", "regime": regime, "epoch": np.nan, "train_loss": np.nan, "dev_eer": dev_eer,
            "n_train": len(idx), "seconds": time.perf_counter() - t0}


def train_seqnet(store: FeatureStore, dev: FeatureStore, regime: str) -> list[dict]:
    import torch

    torch.manual_seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)
    y_all = label_vector(store.meta).astype(np.float32)
    idx = regime_indices(store, regime)
    stats_sample = np.sort(rng.choice(idx, size=min(20000, len(idx)), replace=False))
    pooled_sample = store.pooled_rows(stats_sample)
    pooled_mean, pooled_std = pooled_sample.mean(axis=0), pooled_sample.std(axis=0) + 1e-6
    model = build_seqnet(pooled_mean, pooled_std)

    d = dev_indices(dev, regime)
    y_dev, dev_seq, dev_pooled = label_vector(dev.meta)[d], dev.seq_rows(d), dev.pooled_rows(d)

    n_fake = y_all[idx].sum()
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(idx) - n_fake) / n_fake))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    log, best_eer, best_state, best_epoch, wait = [], np.inf, None, 0, 0
    t0 = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = rng.permutation(idx)
        losses = []
        for b in range(0, len(perm), BATCH):
            batch = np.sort(perm[b:b + BATCH])
            optimizer.zero_grad()
            loss = loss_fn(
                model(torch.from_numpy(store.seq_rows(batch)), torch.from_numpy(store.pooled_rows(batch))),
                torch.from_numpy(y_all[batch]),
            )
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        dev_eer = eer(y_dev, predict_seqnet(model, dev_seq, dev_pooled))
        log.append({"model": "seq_full", "regime": regime, "epoch": epoch, "train_loss": float(np.mean(losses)),
                    "dev_eer": dev_eer, "n_train": len(idx), "seconds": time.perf_counter() - t0})
        print(f"Sequência · {regime} · época {epoch}: perda {np.mean(losses):.4f}, EER no dev {dev_eer:.3f}", flush=True)
        if dev_eer < best_eer:
            best_eer, best_state, best_epoch, wait = dev_eer, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    torch.save({"state_dict": best_state, "pooled_mean": pooled_mean, "pooled_std": pooled_std,
                "epoch": best_epoch, "dev_eer": best_eer}, seq_model_path(regime))
    print(f"Sequência · {regime}: melhor época {best_epoch}, EER no dev {best_eer:.3f}")
    return log


def main() -> None:
    import torch

    torch.set_num_threads(TORCH_THREADS)
    store, dev = FeatureStore(FULL_DIR), FeatureStore(DEV_DIR)
    print(f"Treino completo: {len(store.meta)} janelas de {store.meta['source_file'].nunique()} gravações; "
          f"dev: {len(dev.meta)} janelas.")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    log = []
    for regime in REGIMES:
        log.append(train_lr(store, dev, regime))
    for regime in REGIMES:
        log.extend(train_seqnet(store, dev, regime))
    pd.DataFrame(log).to_csv(LOG_OUT, index=False)
    print(f"Registro do treino salvo em {LOG_OUT}")


if __name__ == "__main__":
    main()
