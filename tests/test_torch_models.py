"""Classificador de sequência (src/sequence_classifier.py): a rede em PyTorch.

Pulado onde o PyTorch não está instalado (o job de CI com PyTorch roda estes testes)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src import sequence_classifier as sc  # noqa: E402
from src.sequence_features import N_FRAMES, PCA_DIM  # noqa: E402

pytestmark = pytest.mark.torch


@pytest.fixture(autouse=True)
def deterministic_torch():
    torch.manual_seed(0)
    torch.set_num_threads(1)


def new_model():
    return sc.build_seqnet(np.zeros(768, dtype=np.float32), np.ones(768, dtype=np.float32))


def toy_batch(rng, n=128):
    """Duas classes separáveis: a classe 1 tem a primeira componente deslocada."""
    y = rng.integers(0, 2, n)
    seq = rng.normal(0, 1, (n, N_FRAMES, PCA_DIM)).astype(np.float32)
    seq[:, :, 0] += 2.0 * y[:, None]
    pooled = rng.normal(0, 1, (n, 768)).astype(np.float32)
    return seq, pooled, y


def test_has_about_230_thousand_parameters():
    parameters = sum(p.numel() for p in new_model().parameters())
    assert 200_000 < parameters < 260_000  # "~230 mil parâmetros", como diz o artigo


def test_one_logit_per_window(rng):
    model = new_model().eval()
    seq, pooled, _ = toy_batch(rng, n=5)
    with torch.inference_mode():
        logits = model(torch.as_tensor(seq), torch.as_tensor(pooled))
    assert logits.shape == (5,)


def test_predictions_are_probabilities_and_do_not_depend_on_the_batch_size(rng):
    model = new_model()
    seq, pooled, _ = toy_batch(rng, n=40)
    small = sc.predict_seqnet(model, seq, pooled, batch=7)
    large = sc.predict_seqnet(model, seq, pooled, batch=64)
    assert small.shape == (40,) and np.all((small >= 0) & (small <= 1))
    np.testing.assert_allclose(small, large, atol=1e-5)


def test_predicting_nothing_returns_an_empty_array():
    empty = sc.predict_seqnet(new_model(), np.zeros((0, N_FRAMES, PCA_DIM), np.float32), np.zeros((0, 768), np.float32))
    assert empty.shape == (0,)


@pytest.fixture
def trained(rng):
    model = new_model()
    seq, pooled, y = toy_batch(rng, n=256)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    inputs = (torch.as_tensor(seq), torch.as_tensor(pooled))
    target = torch.as_tensor(y, dtype=torch.float32)
    model.train()
    losses = []
    for _ in range(40):
        optimizer.zero_grad()
        loss = loss_fn(model(*inputs), target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return model.eval(), (seq, pooled, y), losses


def test_learns_a_separable_problem(trained):
    model, (seq, pooled, y), losses = trained
    assert losses[-1] < 0.6 * losses[0]
    accuracy = ((sc.predict_seqnet(model, seq, pooled) >= 0.5).astype(int) == y).mean()
    assert accuracy > 0.9


def test_checkpoint_round_trip_keeps_the_predictions(trained, tmp_path, monkeypatch):
    model, (seq, pooled, _), _ = trained
    monkeypatch.setattr(sc, "MODELS_DIR", tmp_path)
    torch.save(
        {"pooled_mean": np.zeros(768, dtype=np.float32), "pooled_std": np.ones(768, dtype=np.float32),
         "state_dict": model.state_dict()},
        sc.seq_model_path("clean"),
    )
    loaded = sc.load_seqnet("clean")
    np.testing.assert_allclose(
        sc.predict_seqnet(loaded, seq, pooled), sc.predict_seqnet(model, seq, pooled), atol=1e-6,
    )


@pytest.mark.filterwarnings("ignore::DeprecationWarning", "ignore::UserWarning")
def test_int8_quantization_keeps_the_decisions(trained):
    # O teste de latência do projeto usa a rede quantizada em 8 bits.
    model, (seq, pooled, _), _ = trained
    quantized = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    reference = sc.predict_seqnet(model, seq, pooled)
    approx = sc.predict_seqnet(quantized, seq, pooled)
    assert ((reference >= 0.5) == (approx >= 0.5)).mean() >= 0.95
    assert np.max(np.abs(reference - approx)) < 0.1
