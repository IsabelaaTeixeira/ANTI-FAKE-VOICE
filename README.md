# ANTI-FAKE-VOICE: Detecção de Deepfakes de Voz em Streaming na Borda (MEC) e Robustez ao Codec 5G (VoNR/EVS)

Projeto de pesquisa acadêmica que investiga uma pergunta simples: **um classificador
de voz clonada erra mais quando o áudio passa por um codec de telecom (GSM, VoLTE,
VoNR/5G-EVS) — do mesmo jeito que ouvintes humanos erram mais, segundo um artigo de
referência** — e se **Data Augmentation** (treinar o modelo já expondo-o a esses canais
distorcidos) resolve essa queda de desempenho. Tudo isso sob a restrição de latência
de um nó MEC (Multi-access Edge Computing): o classificador precisa decidir mais rápido
do que a duração da própria janela de áudio, para caber num cenário de detecção em
tempo real numa ligação.

Achado central do projeto: a compressão por codec de fato prejudica o classificador
"limpo" (mais no VoNR do que nos outros dois, confirmando a hipótese do artigo de
referência), e Data Augmentation resolve isso — mas o modelo continua falhando ao ser
testado numa voz, microfone ou ambiente que nunca apareceu no treino, mesmo depois de
retreinado com uma única sessão de gravação extra. Essa segunda parte (generalização
para pessoas/gravações novas) é um problema diferente do problema do canal, e ficou
documentada como achado, não escondida. Os detalhes de cada etapa, com os números,
estão nas tabelas de `results/` e no relatório técnico completo (gerado durante o
desenvolvimento, fora deste repositório).

## As frentes do projeto

O projeto tem 5 frentes, cada uma com seu próprio script em `src/`:

1. **Simulação de canal** (`channel_sim.py`) — recodifica um áudio limpo como se
   tivesse passado por uma ligação GSM, VoLTE ou VoNR/5G, usando os codecs de verdade
   que essas redes usam (não é só adicionar ruído: é o mesmo algoritmo de compressão
   com perdas que uma operadora usa).
2. **Extração de atributos** (`features.py`) — transforma cada 1 segundo de áudio em
   números que um classificador consegue usar: um jeito leve e rápido (MFCC, 26
   números) e um jeito pesado e mais rico (embedding de uma rede neural pré-treinada,
   Wav2Vec2, 768 números).
3. **Treino e avaliação com separação por locutor** (`train.py` + `evaluate.py`) —
   treina os classificadores garantindo que nenhum locutor de teste tenha aparecido no
   treino (Leave-One-Speaker-Out), e mede o desempenho com Acurácia, F1 e EER
   (Equal Error Rate) por canal.
4. **Latência** (`latency_bench.py`) — cronometra cada arquitetura para saber se ela
   cabe no orçamento de tempo de uma ligação ao vivo.
5. **Testes com a própria voz** (`clone_voice.py`, `live_demo.py`,
   `speaker_dependent_eval.py`, `own_voice_ceiling.py`,
   `own_voice_pretrain_test.py`) — a parte mais "prática": clona a voz da
   pesquisadora com IA de verdade (XTTS v2) e testa o classificador ao vivo pelo
   microfone, contra a gravação clonada, e antes/depois de um retreino com essa voz.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

O FFmpeg do sistema (via `apt`) **não** vem com os encoders AMR necessários para
simular GSM e VoLTE. É preciso compilar um FFmpeg próprio:

```bash
ffmpeg -encoders | grep -i amr   # tem que listar libopencore_amrnb e libvo_amrwbenc
```

Se faltar, compile com (dependências de build via `apt`, depois `./configure` e
`make`, instalando em `tools/ffmpeg-amr/`):

```bash
--enable-gpl --enable-nonfree --enable-version3 \
--enable-libopencore-amrnb --enable-libvo-amrwbenc
```

Para o canal VoNR (5G), o projeto usa o **codec de referência EVS** (3GPP TS 26.442/
26.443), compilado a partir do código de referência da 3GPP em `tools/evs-src/`
(binários `EVS_cod`/`EVS_dec`). Confira se os dois setups estão prontos com:

```bash
python -m src.channel_sim
```

Para os testes de clonagem de voz (`clone_voice.py`), o Coqui TTS exige uma versão de
`transformers` incompatível com o resto do projeto — por isso ele roda num ambiente
virtual **separado**:

```bash
python3 -m venv .venv-tts
.venv-tts/bin/pip install torch torchaudio "coqui-tts[codec]"
```

## Estrutura de pastas

```
data/raw/KAGGLE/       # dataset DEEP-VOICE original (baixado manualmente do Kaggle), não versionado
data/manifest.csv       # tabela mestre: filepath, speakers, label, channel — gerada por build_dataset.py
data/features_*.csv/.npz  # atributos extraídos — gerados por features.py, não versionados (regeneráveis)
data/my_voice_reference.wav / cloned_voice.wav  # voz da pesquisadora, real e clonada — não versionados
src/                    # todo o pipeline, um script por etapa (ver seção abaixo)
models/                 # modelos treinados (.joblib) — não versionados, regeneráveis
results/                # tabelas finais (.csv) — as pequenas ficam versionadas, cv_predictions.csv (18MB) não
artigos/                # literatura de referência; só o índice REFS.txt é versionado (PDFs têm direitos autorais de terceiros)
tools/                  # FFmpeg custom e código de referência EVS compilados — não versionados
```

## Passo a passo: treinar, testar e usar cada ferramenta

Os scripts foram feitos para rodar nesta ordem (cada um lê o que o anterior gerou):

### 1. Baixar o dataset
Baixe o **DEEP-VOICE** (Kaggle, Jordan J Bird) e extraia em `data/raw/KAGGLE/AUDIO/`,
separado em `REAL/` e `FAKE/`.

### 2. Montar o dataset áudio × canal
Fatia cada áudio em janelas de 1 segundo e aplica os canais escolhidos:

```bash
python -m src.build_dataset --channels clean gsm volte vonr
```

Por decisão do projeto (não técnica), os locutores `musk` e `trump` são excluídos
automaticamente (ver `EXCLUDED_SPEAKERS` em `build_dataset.py`). Gera
`data/manifest.csv`.

### 3. Extrair os atributos

```bash
python -m src.features --kind mfcc
python -m src.features --kind wav2vec2   # mais lento; retoma sozinho se for interrompido
```

### 4. Treinar com validação cruzada Leave-One-Speaker-Out

```bash
python -m src.train
```

Treina os 4 modelos (MFCC/Wav2Vec2 × canal-limpo/canal-augmentado), sempre testando
num locutor que ficou de fora do treino. Salva as predições em
`results/cv_predictions.csv`.

### 5. Gerar a tabela de avaliação (Acurácia, F1, EER por canal)

```bash
python -m src.evaluate
```

Gera `results/channel_model_table.csv`.

### 6. Medir a latência (MFCC+RF vs. Wav2Vec2+LR)

```bash
python -m src.latency_bench
```

Gera `results/latency.csv`, comparando com o orçamento de 150ms de uma ligação ao
vivo (ITU-T G.114) e com a duração da própria janela (1000ms).

### 7. Testar ao vivo pelo microfone

```bash
python -m src.live_demo --seconds 15
# ou, com um modelo específico:
python -m src.live_demo --seconds 15 --model models/mfcc_rf_with_own_voice.joblib
```

Grava em janelas de 1s via `arecord` (ALSA) e classifica cada uma em tempo real,
mostrando a probabilidade de ser clonada. **Rode este comando você mesma no seu
próprio terminal** — como ele espera sua fala em tempo real, rodar por uma ferramenta
automatizada não permite reagir à contagem regressiva.

### 8. Clonar a própria voz com IA de verdade (XTTS v2)

```bash
.venv-tts/bin/python src/clone_voice.py
```

Requer `data/my_voice_reference.wav` (uma gravação de referência sua). Gera
`data/cloned_voice.wav`. Roda no ambiente virtual separado `.venv-tts/` (ver Setup).
Ao usar, você concorda com a licença CPML da Coqui (uso livre para pesquisa/não
comercial).

### 9. Testar um locutor que já estava no treino (referência de "teto")

```bash
python -m src.speaker_dependent_eval
```

Avalia o modelo final em locutores que ele já viu durante o treino (Taylor Swift,
Biden, Ryan) — não é um teste de generalização, é uma referência de quanto o
desempenho cai quando o locutor é desconhecido (LOSO) vs. conhecido.

### 10. Testar a própria voz contra o modelo original (sem retreinar)

```bash
python -m src.own_voice_pretrain_test
```

Mesma metodologia do passo 9, mas com a voz da pesquisadora (real + clonada) contra o
modelo que nunca a viu — usando os mesmos 4 canais simulados.

### 11. Retreinar incluindo a própria voz, e reavaliar

```bash
python -m src.own_voice_ceiling
```

Junta a voz da pesquisadora (real + clonada, nos 4 canais) ao dataset e retreina,
salvando `models/mfcc_rf_with_own_voice.joblib`. Avalia no que acabou de entrar no
treino (referência de teto, igual ao passo 9) — depois, repita o passo 7 com
`--model models/mfcc_rf_with_own_voice.joblib` para ver se isso generaliza para uma
gravação *nova*, fora do que entrou no retreino.

## Resultados

As tabelas pequenas ficam versionadas em `results/`:

| Arquivo | O que mostra |
|---|---|
| `channel_model_table.csv` | Acurácia/F1/EER por canal × modelo × regime (limpo vs. augmentado) — resultado central do projeto |
| `latency.csv` | Latência (média/mediana/p95) de cada arquitetura, vs. os orçamentos de tempo |
| `speaker_dependent_table.csv` | Teto de desempenho em locutores conhecidos (Taylor/Biden/Ryan) |
| `own_voice_pretrain_table.csv` | Voz própria contra o modelo original (antes de qualquer retreino) |
| `own_voice_ceiling_table.csv` | Voz própria depois de incluída no retreino |

(`cv_predictions.csv`, as predições janela-a-janela usadas para montar
`channel_model_table.csv`, não é versionado por ser grande — é regenerado pelo
passo 4.)

## Limitações e decisões metodológicas

- **Locutores excluídos por decisão do projeto**: `musk` e `trump` foram removidos do
  dataset DEEP-VOICE por decisão pessoal da pesquisadora, não por motivo técnico —
  isso reduz um pouco a quantidade de dados disponível.
- **EER em vez de só acurácia**: o dataset é desbalanceado entre classes/canais, e
  acurácia crua esconde isso; EER é o padrão em biometria/detecção de fraude.
- **Amostra pequena para a voz própria** (~15s reais + ~9s clonadas, antes de aplicar
  canal): os números dos testes com a voz da pesquisadora são ilustrativos, não
  estatisticamente equivalentes aos ~centenas/milhares de janelas por locutor do
  dataset original.
- **Retreino de uma sessão só tende a decorar, não generalizar**: incluir a voz da
  pesquisadora no retreino chegou a 100% de acurácia nos dados usados no retreino, mas
  falhou numa gravação nova dela mesma — evidência de que o modelo memorizou aquela
  sessão específica de gravação, não aprendeu uma representação generalizável da voz.
  Isso é reportado como achado/limitação, não escondido.

## Referências

A relação completa dos artigos usados e o papel de cada um está em
[`artigos/REFS.txt`](artigos/REFS.txt). Os PDFs em si não são versionados neste
repositório por terem direitos autorais de terceiros.
