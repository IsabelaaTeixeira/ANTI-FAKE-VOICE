# Atalhos para o dia a dia. Tudo aqui também funciona digitando o comando.
PYTHON ?= python

.PHONY: test test-torch lint docker-build docker-test

test:            ## testes do núcleo (sem PyTorch e sem os codecs)
	$(PYTHON) -m pytest -m "not torch and not codecs"

test-torch:      ## testes da rede do classificador de sequência (precisa do PyTorch)
	$(PYTHON) -m pytest tests/test_torch_models.py tests/test_hygiene.py

lint:            ## erros de sintaxe e de lógica
	$(PYTHON) -m ruff check .

docker-build:    ## imagem com FFmpeg AMR e o codec EVS compilados
	docker build --target test -t anti-fake-voice .

docker-test: docker-build   ## testes de integração dos codecs dentro da imagem
	docker run --rm anti-fake-voice python -m pytest -m codecs -p no:cacheprovider
