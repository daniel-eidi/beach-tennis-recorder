# 🎾 Beach Tennis Recorder — CLAUDE.md

> Guia de desenvolvimento multi-agente para o app de gravação e análise de jogadas de beach tênis.
> Este arquivo é o contrato de trabalho entre todos os agentes do projeto.

---

## 🏗️ Visão Geral do Projeto

**Objetivo:** App mobile (iOS + Android via Flutter) que grava partidas de beach tênis, detecta automaticamente rallies usando visão computacional (YOLOv8), e gera clipes de vídeo de cada ponto.

**Stack:**
- Mobile: Flutter (Dart)
- Visão Computacional: YOLOv8 (Python, TFLite/ONNX para mobile)
- Processamento de Vídeo: FFmpeg
- Backend: FastAPI (Python)
- Storage: Google Cloud Storage (GCS) ou local
- CI/CD iOS: Codemagic

---

## 🤖 Agentes e Responsabilidades

### AGENT-01 · Mobile (Flutter)
Responsável pela UI, câmera, buffer circular e biblioteca de clipes.

### AGENT-02 · Vision (YOLOv8)
Responsável pelo dataset, treinamento do modelo, exportação TFLite/ONNX e lógica de detecção de rally.

### AGENT-03 · Video (FFmpeg)
Responsável pelo processamento de vídeo: corte, naming, compressão e export de clipes.

### AGENT-04 · Backend (FastAPI)
Responsável pela API REST, storage GCS, autenticação e endpoints de upload/download.

### AGENT-05 · QA & Integration
Responsável pelos testes de integração entre agentes, testes em campo e validação do fluxo completo.

### AGENT-00 · Orchestrator
Coordena os demais agentes, resolve conflitos de interface, mantém o contrato de APIs entre camadas.

---

## 📁 Estrutura de Pastas

```
beach-tennis-recorder/
├── CLAUDE.md                    ← este arquivo
├── mobile/                      ← AGENT-01
│   ├── lib/
│   │   ├── main.dart
│   │   ├── screens/
│   │   │   ├── home_screen.dart
│   │   │   ├── recording_screen.dart
│   │   │   └── library_screen.dart
│   │   ├── services/
│   │   │   ├── camera_service.dart
│   │   │   ├── buffer_service.dart
│   │   │   ├── clip_service.dart
│   │   │   └── rally_controller.dart
│   │   └── models/
│   │       ├── rally.dart
│   │       └── clip.dart
│   ├── assets/models/           ← modelo TFLite exportado por AGENT-02
│   └── pubspec.yaml
│
├── vision/                      ← AGENT-02
│   ├── dataset/
│   │   ├── images/
│   │   ├── labels/
│   │   └── data.yaml
│   ├── train.py
│   ├── export_tflite.py
│   ├── export_onnx.py
│   ├── inference_test.py
│   └── models/
│       ├── best.pt              ← PyTorch
│       ├── best.tflite          ← para mobile
│       └── best.onnx            ← alternativo
│
├── video/                       ← AGENT-03
│   ├── clip_processor.py
│   ├── ffmpeg_wrapper.py
│   ├── naming_convention.py
│   └── tests/
│
├── backend/                     ← AGENT-04
│   ├── main.py
│   ├── routers/
│   │   ├── clips.py
│   │   ├── matches.py
│   │   └── upload.py
│   ├── services/
│   │   ├── gcs_service.py
│   │   └── auth_service.py
│   ├── models/
│   │   └── schemas.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── tests/                       ← AGENT-05
│   ├── integration/
│   ├── field_tests/
│   └── mock_videos/
│
└── docs/
    ├── api_contract.md          ← contrato de APIs entre agentes
    ├── rally_states.md
    └── calibration_guide.md
```

---

## 🔗 Contrato de APIs entre Agentes

### AGENT-02 → AGENT-01 (modelo exportado)
```
Arquivo: mobile/assets/models/ball_detector.tflite
Input:   [1, 640, 640, 3] float32 (RGB normalizado 0-1)
Output:  [1, 25200, 6] float32 (x, y, w, h, confidence, class)
Classes: {0: "ball", 1: "net", 2: "court_line"}
```

### AGENT-01 → AGENT-03 (trigger de clipe)
```dart
// RallyEvent enviado ao ClipProcessor
class RallyEvent {
  final DateTime startTime;    // T-3s antes do saque
  final DateTime endTime;      // momento do fim do rally
  final String bufferFilePath; // path do buffer .mp4
  final int matchId;
  final int rallyNumber;
}
```

### AGENT-03 → AGENT-04 (upload de clipe)
```
POST /api/v1/clips/upload
Content-Type: multipart/form-data
Body: {
  file: <binary .mp4>,
  match_id: int,
  rally_number: int,
  duration_seconds: float,
  timestamp_start: ISO8601
}
Response: { clip_id: str, gcs_url: str, thumbnail_url: str }
```

### AGENT-04 → AGENT-01 (listagem de clipes)
```
GET /api/v1/clips?match_id={id}
Response: [{ clip_id, rally_number, duration, thumbnail_url, stream_url }]
```

---

## 🔄 Máquina de Estados — Rally

```
                   bola detectada + velocidade > threshold
IDLE ─────────────────────────────────────────────────────► EM_JOGO
  ▲                                                              │
  │   salva clipe                                               │ bola cruza rede
  │   reseta buffer      FIM_RALLY ◄───────────────────────────┘
  └──────────────────────────┤
                             │ triggers:
                             │  - bola.tocou_chao == True
                             │  - bola.fora_limite == True
                             │  - timeout > 8s sem detecção
```

### Constantes da Máquina de Estados
```python
BUFFER_PRE_RALLY_SECONDS = 3      # segundos antes do saque a capturar
BUFFER_POST_RALLY_SECONDS = 2     # segundos após fim do rally
VELOCITY_THRESHOLD = 15.0         # px/frame para considerar bola em movimento
CONFIDENCE_THRESHOLD = 0.45       # confiança mínima do modelo
RALLY_TIMEOUT_SECONDS = 8         # tempo máximo sem detectar bola antes de encerrar
NET_CROSS_REQUIRED = True         # bola deve cruzar a rede para confirmar rally
```

---

## 📋 Tasks por Agente

### AGENT-01 · Mobile Flutter

```
TASK-01-01  Setup projeto Flutter com camera_plugin e ffmpeg_kit
TASK-01-02  Implementar CameraService com preview 1080p/30fps
TASK-01-03  Implementar BufferService (buffer circular de 60s em disco)
TASK-01-04  Implementar RallyController (máquina de estados)
TASK-01-05  Integrar modelo TFLite via tflite_flutter
TASK-01-06  Implementar detecção de frame em isolate separado
TASK-01-07  Implementar ClipService (trigger → salvar segmento do buffer)
TASK-01-08  Tela HomeScreen
TASK-01-09  Tela RecordingScreen (preview + indicador de rally + contador)
TASK-01-10  Tela LibraryScreen (lista de clipes por partida)
TASK-01-11  Player de vídeo inline para revisão de clipes
TASK-01-12  Função de compartilhamento de clipes (Share sheet)
TASK-01-13  Calibração de quadra (marcar 4 cantos na tela → homografia)
TASK-01-14  Configurações (resolução, FPS, buffer size, threshold)
```

### AGENT-02 · Vision YOLOv8

```
TASK-02-01  Definir protocolo de coleta de vídeos para dataset
TASK-02-02  Gravar/coletar 30+ min de beach tênis em diferentes condições
TASK-02-03  Extrair frames a 5fps para anotação (~3000 imagens)
TASK-02-04  Anotar dataset no Roboflow (bola, rede, linha da quadra)
TASK-02-05  Configurar data.yaml com classes e splits (70/20/10)
TASK-02-06  Fine-tuning YOLOv8n com dataset anotado
TASK-02-07  Avaliar métricas: mAP@0.5 > 0.75 como meta mínima
TASK-02-08  Implementar lógica de tracking de bola entre frames (ByteTrack)
TASK-02-09  Implementar detecção de quique (mudança brusca de direção Y)
TASK-02-10  Implementar detecção de cruzamento de rede (zona central)
TASK-02-11  Exportar modelo para TFLite (INT8 quantizado para mobile)
TASK-02-12  Exportar modelo para ONNX (fallback Android)
TASK-02-13  Benchmark de performance: alvo < 50ms/frame no iPhone 12+
TASK-02-14  Documentar thresholds recomendados por condição de luz
```

### AGENT-03 · Video Processing

```
TASK-03-01  Wrapper Python para FFmpeg com interface clara
TASK-03-02  Função cut_clip(input, start_ts, end_ts, output) sem re-encoding
TASK-03-03  Convenção de naming: rally_{match_id}_{number}_{date}_{time}.mp4
TASK-03-04  Geração de thumbnail (frame do meio do clipe)
TASK-03-05  Compressão opcional H.264/H.265 para upload
TASK-03-06  Fila assíncrona de processamento (não bloquear câmera)
TASK-03-07  Validação de clipe gerado (duração mínima 1s, tamanho máximo 200MB)
TASK-03-08  Testes unitários com vídeos mock
```

### AGENT-04 · Backend FastAPI

```
TASK-04-01  Setup FastAPI + estrutura de routers
TASK-04-02  Modelo de dados: Match, Rally, Clip (Pydantic + Firestore)
TASK-04-03  Endpoint POST /clips/upload (multipart, salva no GCS)
TASK-04-04  Endpoint GET /clips?match_id= (lista com URLs assinadas)
TASK-04-05  Endpoint POST /matches (criar partida, retorna match_id)
TASK-04-06  Serviço GCS (upload, download, signed URLs com TTL 1h)
TASK-04-07  Autenticação simples via API Key (MVP)
TASK-04-08  Dockerfile + deploy no Cloud Run
TASK-04-09  Health check endpoint GET /health
TASK-04-10  Testes de carga básicos (10 uploads simultâneos)
```

### AGENT-05 · QA & Integration

```
TASK-05-01  Criar suite de vídeos mock para testes (rallies reais e borda)
TASK-05-02  Teste integração: câmera → buffer → trigger → clipe
TASK-05-03  Teste integração: clipe → upload → API → GCS → listagem
TASK-05-04  Teste de campo: 1 partida completa, validar % de rallies detectados
TASK-05-05  Medir false positives (ações que não eram rally mas triggeraram)
TASK-05-06  Medir false negatives (rallies reais não detectados)
TASK-05-07  Teste de performance: consumo de bateria em 1h de gravação
TASK-05-08  Teste em diferentes condições: dia claro, nublado, noturno
TASK-05-09  Relatório de bugs com prioridade P0/P1/P2
TASK-05-10  Critérios de aceite MVP: >85% recall de rallies, <10% false positive
```

---

## 🚦 Ordem de Execução e Dependências

```
Sprint 1 (semanas 1-2): Fundação
  AGENT-01: TASK-01-01, 01-02, 01-03          → buffer circular funcionando
  AGENT-02: TASK-02-01, 02-02, 02-03, 02-04   → dataset coletado e anotado
  AGENT-04: TASK-04-01, 04-02, 04-05, 04-09   → API básica rodando

Sprint 2 (semanas 3-4): Modelo + Video
  AGENT-02: TASK-02-05 → 02-11                → modelo TFLite exportado
  AGENT-03: TASK-03-01 → 03-06                → clip processor funcionando
  AGENT-04: TASK-04-03, 04-04, 04-06, 04-07   → upload/download completo

Sprint 3 (semanas 5-6): Integração Mobile
  AGENT-01: TASK-01-04 → 01-12                → app completo integrado
  AGENT-05: TASK-05-01 → 05-03                → testes de integração

Sprint 4 (semanas 7-8): Campo + Ajuste
  AGENT-05: TASK-05-04 → 05-10                → testes reais em quadra
  AGENT-02: ajuste de thresholds baseado em campo
  AGENT-01: TASK-01-13, 01-14                 → calibração e configurações
```

---

## ⚙️ Regras Gerais para Todos os Agentes

1. **Sempre commitar com prefixo do agente:** `[AGENT-02] feat: export tflite model`
2. **Nunca quebrar o contrato de APIs** sem aprovação do AGENT-00
3. **Código deve ter cobertura mínima de 60%** de testes unitários
4. **Logs estruturados** em JSON: `{"agent": "02", "task": "02-11", "status": "ok", "ms": 42}`
5. **Feature flags** para funcionalidades experimentais (ex: `NET_CROSS_REQUIRED`)
6. **Sem dependências cruzadas diretas** entre agentes — apenas via contratos definidos acima

---

## 🧪 Critérios de Aceite do MVP

| Critério | Meta |
|---|---|
| Recall de rallies detectados | ≥ 85% |
| False positive rate | ≤ 10% |
| Latência de detecção | ≤ 50ms/frame |
| Tempo de geração do clipe | ≤ 5s após fim do rally |
| Consumo de bateria (1h gravação) | ≤ 40% da bateria |
| Tamanho médio do clipe | ≤ 30MB |
| Cobertura de testes | ≥ 60% por módulo |

---

## 🔧 Configuração do Ambiente de Desenvolvimento

```bash
# Clone e setup
git clone https://github.com/seu-org/beach-tennis-recorder
cd beach-tennis-recorder

# Backend (Python 3.11+)
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload

# Vision (Python 3.11+)
cd vision
pip install ultralytics roboflow opencv-python
python train.py --config dataset/data.yaml --model yolov8n.pt

# Mobile (Flutter 3.x)
cd mobile
flutter pub get
flutter run

# Variáveis de ambiente necessárias
# .env
GCS_BUCKET_NAME=beach-tennis-clips
GCS_PROJECT_ID=seu-projeto-gcp
API_KEY=sua-chave-de-api
CODEMAGIC_API_TOKEN=token-para-build-ios
```

---

## 📌 Notas para o Orchestrator (AGENT-00)

- O **ponto de maior risco técnico** é a qualidade do dataset (TASK-02-02 a 02-04). Priorizar coleta de dados em campo com variação de iluminação desde o início.
- A **calibração de quadra** (TASK-01-13) deve ser projetada para ser feita uma única vez por local, salva localmente.
- O **buffer circular** deve ser implementado em disco (não memória) para suportar 60s de 1080p sem crashar em dispositivos com RAM limitada.
- O **processamento de frames** deve rodar em `Isolate` separado no Flutter para não travar a UI.
- Para o MVP, **offline-first**: tudo funciona sem backend, com sync opcional quando tiver rede.
