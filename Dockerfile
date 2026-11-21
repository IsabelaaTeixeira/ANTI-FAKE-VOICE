# syntax=docker/dockerfile:1
#
# Imagem do ANTI-FAKE-VOICE: compila a cadeia de codecs de telefonia (FFmpeg com
# AMR-NB/AMR-WB e o codec de referencia EVS da 3GPP) e instala o projeto.
#
# Alvos (docker build --target <alvo>):
#   test      cadeia de codecs + dependencias de teste. Uso:
#               docker build --target test -t anti-fake-voice .
#               docker run --rm anti-fake-voice                    # confere os codecs
#               docker run --rm anti-fake-voice python -m pytest -m codecs
#   pipeline  o alvo test + PyTorch (CPU) e as demais dependencias do pipeline,
#             menos o coqui-tts, que precisa de um ambiente separado (ver README).
#
# Passe sempre o --target. Sem ele, o docker build constroi o ultimo estagio
# (pipeline, com o PyTorch); e o construtor antigo do Docker, sem o BuildKit,
# constroi todos os estagios anteriores ao pedido, por isso o test vem antes.
#
# O codigo do EVS e o FFmpeg sao baixados das fontes oficiais, com checksum.

# ----------------------------------------------------------------- FFmpeg com AMR
# O FFmpeg do apt nao traz os encoders AMR (libopencore_amrnb e libvo_amrwbenc).
FROM debian:bookworm-slim AS ffmpeg-build
ARG FFMPEG_VERSION=7.1.1
ARG FFMPEG_SHA256=733984395e0dbbe5c046abda2dc49a5544e7e0e1e2366bba849222ae9e3a03b1
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential nasm pkg-config curl ca-certificates xz-utils \
        libopencore-amrnb-dev libvo-amrwbenc-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
RUN curl -fsSL -o ffmpeg.tar.xz "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz" \
    && echo "${FFMPEG_SHA256}  ffmpeg.tar.xz" | sha256sum -c - \
    && tar -xJf ffmpeg.tar.xz --strip-components=1
RUN ./configure \
        --prefix=/opt/ffmpeg-amr \
        --disable-autodetect --disable-debug --disable-doc \
        --disable-ffplay --disable-ffprobe \
        --enable-gpl --enable-version3 \
        --enable-libopencore-amrnb --enable-libvo-amrwbenc \
    && make -j"$(nproc)" \
    && make install

# ------------------------------------------------- codec de referencia EVS (3GPP)
# TS 26.443 v13.3.0 (ANSI C, ponto flutuante). O Makefile gera EVS_cod e EVS_dec.
FROM debian:bookworm-slim AS evs-build
ARG EVS_ZIP_URL=https://www.3gpp.org/ftp/Specs/archive/26_series/26.443/26443-d30.zip
ARG EVS_ZIP_SHA256=4b01eab3f3bbcc9df756e840c320b3374911b49fe16af4ce0f14ed8c9ea17b8f
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl ca-certificates unzip \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
RUN curl -fsSL -o evs.zip "${EVS_ZIP_URL}" \
    && echo "${EVS_ZIP_SHA256}  evs.zip" | sha256sum -c - \
    && unzip -q evs.zip '*ANSI-C_source_code.zip' \
    && unzip -q ./*ANSI-C_source_code.zip
WORKDIR /build/c-code
RUN make -j"$(nproc)" RELEASE=1 DEBUG=0

# ------------------------------------------------------------ base com os codecs
FROM python:3.12-slim-bookworm AS base
RUN apt-get update \
    && apt-get install -y --no-install-recommends libopencore-amrnb0 libvo-amrwbenc0 \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
# Os caminhos abaixo sao os que src/channel_sim.py espera.
COPY --from=ffmpeg-build /opt/ffmpeg-amr/bin/ffmpeg tools/ffmpeg-amr/bin/ffmpeg
COPY --from=evs-build /build/c-code/EVS_cod /build/c-code/EVS_dec tools/evs-src/

# ------------------------------------------------------- alvo test: imagem leve
FROM base AS test
COPY requirements-test.txt ./
RUN pip install -r requirements-test.txt
COPY pyproject.toml requirements.txt README.md .gitignore ./
COPY src src
COPY tests tests
COPY results results
RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app
CMD ["python", "-m", "src.channel_sim"]

# ------------------------------------------------ alvo pipeline: pipeline completo
# Fica por ultimo: e o que sai de um docker build sem --target.
FROM test AS pipeline
USER root
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && grep -v -i coqui requirements.txt > /tmp/requirements-pipeline.txt \
    && pip install -r /tmp/requirements-pipeline.txt
USER app
