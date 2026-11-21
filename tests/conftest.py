"""Fixtures comuns dos testes."""
import numpy as np
import pytest


@pytest.fixture
def rng():
    """Gerador de números aleatórios com semente fixa: os testes são determinísticos."""
    return np.random.default_rng(1234)
