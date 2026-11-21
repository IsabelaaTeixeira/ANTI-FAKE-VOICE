"""Higiene do repositório: todo módulo importa sem efeitos colaterais, as
dependências estão declaradas e não há caminhos de máquina, segredos nem
gravações pessoais versionados."""
import ast
import importlib
import pkgutil
import re
import subprocess
import sys
import traceback
from pathlib import Path

import pytest

import src

ROOT = Path(__file__).resolve().parent.parent
# Dependências pesadas que o ambiente de teste leve não instala.
OPTIONAL_DEPENDENCIES = {"torch", "matplotlib", "TTS"}
IMPORT_TO_REQUIREMENT = {"sklearn": "scikit-learn", "TTS": "coqui-tts"}
MODULES = sorted(m.name for m in pkgutil.iter_modules(src.__path__))


def broken_optional_dependency(error):
    """A dependência opcional em que o erro de importação aconteceu, se foi dentro de uma.

    O coqui-tts, por exemplo, roda num ambiente próprio (.venv-tts, veja o README),
    porque exige outra versão do transformers: num ambiente que tenha os dois, ele
    está instalado mas não importa, e isso não é defeito do módulo de src/."""
    for frame, _ in traceback.walk_tb(error.__traceback__):
        parts = Path(frame.f_code.co_filename).parts
        for name in OPTIONAL_DEPENDENCIES:
            if name in parts:
                return name
    return None


@pytest.mark.parametrize("module", MODULES)
def test_module_imports_without_running_anything(module):
    try:
        importlib.import_module(f"src.{module}")
    except ModuleNotFoundError as error:
        if error.name in OPTIONAL_DEPENDENCIES:
            pytest.skip(f"{error.name} não está instalado neste ambiente")
        raise
    except ImportError as error:
        dependency = broken_optional_dependency(error)
        if dependency:
            pytest.skip(f"{dependency} está instalado mas não importa neste ambiente: {error}")
        raise


def declared_requirements():
    names = set()
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            names.add(re.split(r"[=<>~!\[;\s]", line)[0].lower().replace("_", "-"))
    return names


def imported_third_party():
    found = {}
    for path in sorted((ROOT / "src").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            else:
                continue
            for name in names:
                if name not in sys.stdlib_module_names and name != "src":
                    found.setdefault(name, set()).add(path.name)
    return found


def test_every_third_party_import_is_declared_in_requirements():
    declared = declared_requirements()
    missing = {
        name: sorted(files)
        for name, files in imported_third_party().items()
        if IMPORT_TO_REQUIREMENT.get(name, name).lower() not in declared
    }
    assert not missing, f"importadas em src/ mas ausentes de requirements.txt: {missing}"


MACHINE_PATHS = [re.compile(r"/home/\w+"), re.compile(r"/Users/\w+"), re.compile(r"[A-Za-z]:\\Users")]
SECRETS = [
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)(api[_-]?key|secret|password)\s*=\s*[\"'][^\"']{8,}[\"']"),
]


def tracked_text_files():
    paths = list((ROOT / "src").glob("*.py")) + [ROOT / "README.md", ROOT / "requirements.txt"]
    return [p for p in paths if p.exists()]


@pytest.mark.parametrize("path", tracked_text_files(), ids=lambda p: p.name)
def test_no_machine_paths_or_secrets(path):
    text = path.read_text(encoding="utf-8")
    for pattern in MACHINE_PATHS + SECRETS:
        assert not pattern.search(text), f"{path.name} casa com {pattern.pattern}"


def test_gitignore_keeps_personal_voice_recordings_out_of_the_repository():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "my_voice_reference" in text and "cloned_voice" in text


def test_no_audio_or_model_binaries_are_tracked():
    try:
        files = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("não é um clone do git")
    heavy = [f for f in files if f.lower().endswith((".wav", ".flac", ".mp3", ".joblib", ".npy", ".npz", ".pt"))]
    assert not heavy, f"arquivos pesados ou pessoais versionados: {heavy}"
