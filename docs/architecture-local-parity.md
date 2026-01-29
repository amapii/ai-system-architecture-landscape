# Enterprise Knowledge AI Platform: Local Parity Architecture

## 1. Positioning

**Deployment Mode:** Local Parity (Cloud-Equivalent, Cost-Minimal)

**Target Audience:**
- Developers learning RAG architecture
- Portfolio and demonstration projects
- Cost-sensitive MVPs and prototypes
- Teams requiring full system understanding
- Air-gapped or privacy-sensitive environments

**Use Cases:**
- Local development and experimentation
- Proof-of-concept validation
- Educational implementations
- Offline-capable knowledge systems
- Privacy-first deployments (no data leaves the machine)

---

## 2. System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENTS                                        │
│                       Browser / CLI / API Client                            │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │ HTTP (localhost)
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DOCKER COMPOSE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    API GATEWAY CONTAINER                            │   │
│  │                    FastAPI + Uvicorn                                │   │
│  │           ┌──────────────────────────────────────────┐              │   │
│  │           │  - Request validation                    │              │   │
│  │           │  - Basic auth (optional)                 │              │   │
│  │           │  - Request routing                       │              │   │
│  │           └──────────────────────────────────────────┘              │   │
│  └─────────────────────────────────────┬───────────────────────────────┘   │
│                                        │                                    │
│          ┌─────────────────────────────┼─────────────────────────┐         │
│          │                             │                         │         │
│          ▼                             ▼                         ▼         │
│  ┌───────────────┐        ┌────────────────────┐    ┌───────────────────┐  │
│  │  INGESTION    │        │  RETRIEVAL         │    │  QUERY            │  │
│  │  CONTAINER    │        │  CONTAINER         │    │  CONTAINER        │  │
│  │               │        │                    │    │                   │  │
│  │  - Parser     │        │  - Query embed     │    │  - Context pack   │  │
│  │  - Chunker    │        │  - Vector search   │    │  - LLM call       │  │
│  │  - Embedder   │        │  - Ranking         │    │  - Response fmt   │  │
│  └───────┬───────┘        └─────────┬──────────┘    └───────────┬───────┘  │
│          │                          │                           │          │
│          │                          │                           │          │
│          ▼                          ▼                           ▼          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    SHARED DATA LAYER                                │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │   │
│  │  │ MinIO       │  │ ChromaDB    │  │ Ollama / vLLM               │  │   │
│  │  │ (S3-compat) │  │ (Vector DB) │  │ (LLM Server)                │  │   │
│  │  │             │  │             │  │                             │  │   │
│  │  │ Documents   │  │ Embeddings  │  │ Llama3 / Mistral / Gemma    │  │   │
│  │  │ Chunks      │  │ Indexes     │  │                             │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                          LOCAL VOLUMES                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ ./data/docs │  │ ./data/     │  │ ./data/     │  │ ./models/           │ │
│  │             │  │   chroma/   │  │   minio/    │  │   llama3/           │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Architecture

### 3.1 Data Storage Layer

| Component | Local Implementation | Specification |
|-----------|---------------------|---------------|
| Raw Document Store | MinIO | S3-compatible, local persistence |
| Processed Chunks | MinIO | JSON format |
| Metadata Store | SQLite / JSON files | Lightweight, file-based |
| Alternative | Local filesystem | Direct file access |

**Deployment:** Docker container (`minio/minio`)
**State:** Persistent (volume-mounted)
**Port:** 9000 (API), 9001 (Console)

**MinIO Configuration:**
```yaml
minio:
  image: minio/minio:latest
  command: server /data --console-address ":9001"
  ports:
    - "9000:9000"
    - "9001:9001"
  volumes:
    - ./data/minio:/data
  environment:
    MINIO_ROOT_USER: minioadmin
    MINIO_ROOT_PASSWORD: minioadmin
```

---

### 3.2 Embedding Service

| Component | Local Implementation | Specification |
|-----------|---------------------|---------------|
| Embedding Model | sentence-transformers | all-MiniLM-L6-v2 (384 dim) |
| Alternative | bge-base-en-v1.5 | 768 dimensions |
| Alternative | nomic-embed-text | 768 dimensions |

**Model Options:**
| Model | Dimensions | Size | Speed | Quality |
|-------|------------|------|-------|---------|
| all-MiniLM-L6-v2 | 384 | 80MB | Fast | Good |
| bge-base-en-v1.5 | 768 | 440MB | Medium | Better |
| nomic-embed-text | 768 | 550MB | Medium | Better |
| e5-large-v2 | 1024 | 1.3GB | Slow | Best |

**State:** Stateless (model loaded in memory)
**Deployment:** Embedded in Ingestion/Retrieval containers

**Python Implementation:**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
```

---

### 3.3 Vector Search

| Component | Local Implementation | Specification |
|-----------|---------------------|---------------|
| Vector Store | ChromaDB | Persistent mode |
| Alternative | FAISS | In-memory + filesystem |
| Alternative | Qdrant | Docker container |

**ChromaDB Configuration:**
```yaml
chroma:
  image: chromadb/chroma:latest
  ports:
    - "8000:8000"
  volumes:
    - ./data/chroma:/chroma/chroma
  environment:
    ANONYMIZED_TELEMETRY: "false"
    PERSIST_DIRECTORY: /chroma/chroma
```

**FAISS Alternative (Embedded):**
```python
import faiss
import numpy as np

# Create index
dimension = 384
index = faiss.IndexFlatIP(dimension)  # Inner product

# Add vectors
index.add(embeddings.astype('float32'))

# Save/Load
faiss.write_index(index, "index.faiss")
index = faiss.read_index("index.faiss")
```

**State:** Persistent (volume-mounted)
**Deployment:** Docker container or embedded in service

---

### 3.4 LLM Inference

| Component | Local Implementation | Specification |
|-----------|---------------------|---------------|
| LLM Server | Ollama | Easy model management |
| Alternative | vLLM | Higher throughput |
| Alternative | llama.cpp | Minimal dependencies |

**Recommended Models:**
| Model | VRAM | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| llama3.2:3b | 4GB | Fast | Good | Development |
| llama3.1:8b | 8GB | Medium | Better | Production |
| mistral:7b | 8GB | Medium | Better | Production |
| gemma2:9b | 10GB | Medium | Better | Production |
| llama3.1:70b | 48GB | Slow | Best | High-quality |

**Ollama Configuration:**
```yaml
ollama:
  image: ollama/ollama:latest
  ports:
    - "11434:11434"
  volumes:
    - ./models:/root/.ollama
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

**vLLM Alternative:**
```yaml
vllm:
  image: vllm/vllm-openai:latest
  command: --model meta-llama/Meta-Llama-3.1-8B-Instruct --dtype auto
  ports:
    - "8080:8000"
  volumes:
    - ./models:/root/.cache/huggingface
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

**State:** Stateless (model loaded in memory)
**Deployment:** Docker container with GPU passthrough

---

### 3.5 API Compatibility Layer

This platform uses the **OpenAI-compatible API contract** as the standard interface for LLM inference across all deployment variants.

**Purpose:**
- Establishes a consistent, well-documented interface for LLM communication
- Enables seamless model and provider substitution without application code changes
- This is a **protocol/interface choice**, not a vendor dependency

**Contract Specification:**
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/chat/completions` | POST | Chat-based generation |
| `/v1/completions` | POST | Text completion |
| `/v1/models` | GET | List available models |

**Benefits:**
- **Model Switching:** Change from Llama to Mistral to Gemma without code changes
- **Provider Substitution:** Swap between Ollama and vLLM transparently
- **Migration Ready:** Same code works when migrating to cloud variants
- **Testing:** Consistent interface for unit and integration tests

**Implementation Requirement:**
- All application services MUST interact with LLMs exclusively via this API contract
- Direct library calls are prohibited in application code
- LLM client abstraction layer handles server-specific configuration

**Local Implementation:**

Ollama natively supports the OpenAI-compatible API:

```python
# shared/clients/llm_client.py
import openai

class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434"):
        self.client = openai.OpenAI(
            base_url=f"{host}/v1",
            api_key="not-needed"  # Ollama doesn't require auth
        )
    
    def chat_completion(self, messages: list, model: str = "llama3.1:8b", **kwargs) -> dict:
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
```

vLLM also provides OpenAI-compatible endpoints by default.

---

### 3.6 LLM Serving Model

**Architecture Pattern:** Long-Running Inference Server

**Key Characteristics:**
- LLMs are deployed as **persistent, long-running Docker containers**
- Models are loaded into GPU memory **once at container startup**
- Models are NOT loaded per request; this would be prohibitively slow
- Application services communicate via HTTP API calls to the inference server

**Request Flow:**
```
Application Service
        │
        │ HTTP POST /v1/chat/completions
        ▼
┌─────────────────────────┐
│ Ollama Container        │
│                         │
│ - Model in GPU memory   │
│ - Responds to requests  │
│ - Persistent process    │
└─────────────────────────┘
```

**Conceptual Parity:**

This self-hosted inference pattern is functionally equivalent to:
- Vertex AI Gemini API (GCP-Native variant)
- vLLM on Cloud Run (Hybrid variant)

The difference is operational: you manage the infrastructure instead of Google.

**Service Characteristics:**
| Aspect | Specification |
|--------|--------------|
| Availability | Depends on local hardware uptime |
| Cold Start | 30-120 seconds (model loading) |
| Scaling | Manual (add GPU, run replicas) |
| Model Loading | On container startup |
| Latency | ~100-500ms typical |

---

### 3.7 Model Lifecycle

**Embedding Models:**
| Aspect | Local Implementation |
|--------|---------------------|
| Weight Storage | `~/.cache/huggingface/` or container layer |
| Loading | On container/service startup |
| Updates | Pull new model version, rebuild container |
| Version Control | Model name in configuration |

**LLM Models:**
| Aspect | Local Implementation |
|--------|---------------------|
| Weight Storage | `./models/` directory or Ollama cache |
| Loading | On container startup (into GPU memory) |
| Updates | `ollama pull <model>` or rebuild vLLM container |
| Version Control | Model tag in configuration |

**Model Update Process (Ollama):**
```bash
# Pull new model version
docker compose exec ollama ollama pull llama3.1:8b

# Restart to load new version
docker compose restart ollama
```

**Model Update Process (vLLM):**
1. Update model name in Dockerfile or docker-compose.yaml
2. Rebuild container: `docker compose build llm`
3. Restart service: `docker compose up -d llm`

**Storage Requirements:**
| Model | Disk Space | Memory (Loaded) |
|-------|------------|----------------|
| llama3.2:3b | ~2GB | ~4GB VRAM |
| llama3.1:8b | ~5GB | ~8GB VRAM |
| mistral:7b | ~4GB | ~8GB VRAM |
| llama3.1:70b | ~40GB | ~48GB VRAM |

**Rollback:**
- Keep previous model weights on disk
- Update configuration to reference previous version
- Restart container

---

### 3.8 Application Services

| Service | Runtime | Responsibility | Resources |
|---------|---------|----------------|-----------|
| API Gateway | Docker (FastAPI) | Auth, routing | 256MB RAM |
| Ingestion Service | Docker (FastAPI) | Document processing | 1GB RAM |
| Retrieval Service | Docker (FastAPI) | Query, search, rank | 1GB RAM |
| Query Service | Docker (FastAPI) | Context pack, LLM call | 512MB RAM |

**FastAPI Service Template:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**State:** Stateless
**Deployment:** Docker Compose

---

### 3.6 Security Layer

| Component | Local Implementation | Configuration |
|-----------|---------------------|---------------|
| Authentication | Basic Auth / API Keys | Environment variables |
| Network | Docker network | Internal service mesh |
| Secrets | .env files | Git-ignored |
| TLS | Optional (Traefik) | Self-signed certs |

**Environment Configuration:**
```bash
# .env file (git-ignored)
API_KEY=your-secret-api-key
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
OLLAMA_HOST=http://ollama:11434
CHROMA_HOST=http://chroma:8000
```

---

### 3.7 Observability

| Component | Local Implementation | Purpose |
|-----------|---------------------|---------|
| Logging | Python logging + stdout | Container logs |
| Monitoring | Prometheus (optional) | Metrics |
| Tracing | Jaeger (optional) | Request tracing |
| UI | Grafana (optional) | Dashboards |

**Minimal Logging Setup:**
```python
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
```

---

## 4. Data Flow

### 4.1 Document Ingestion Flow

```
Document Upload (local file or API)
      │
      ▼
┌─────────────────┐
│ Ingestion       │
│ Service         │
│                 │
│ 1. Parse doc    │  (PyMuPDF, python-docx, etc.)
│ 2. Extract text │
│ 3. Chunk text   │  (overlapping chunks)
│ 4. Clean chunks │
└────────┬────────┘
         │
         ├──────────────────────────────────┐
         │                                  │
         ▼                                  ▼
┌─────────────────┐           ┌────────────────────────┐
│ MinIO           │           │ Embedding Model        │
│                 │           │ (sentence-transformers)│
│ Store chunks    │           │                        │
│ as JSON         │           │ Generate embeddings    │
└─────────────────┘           └────────────┬───────────┘
                                           │
                                           ▼
                              ┌────────────────────────┐
                              │ ChromaDB               │
                              │                        │
                              │ Store vectors          │
                              │ with metadata          │
                              └────────────────────────┘
```

### 4.2 Query Flow

```
User Query (CLI / Web UI / API)
      │
      ▼
┌─────────────────┐
│ API Gateway     │
│                 │
│ Validate        │
│ Route           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Retrieval       │
│ Service         │
│                 │
│ 1. Embed query  │──────▶ Embedding Model
│ 2. Vector search│──────▶ ChromaDB
│ 3. Fetch chunks │──────▶ MinIO (optional)
│ 4. Rank results │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Query Service   │
│                 │
│ 1. Pack context │
│ 2. Build prompt │
│ 3. Call LLM     │──────▶ Ollama (llama3.1:8b)
│ 4. Parse output │
│ 5. Add citations│
└────────┬────────┘
         │
         ▼
    Response (JSON)
```

---

## 5. Deployment Model

### 5.1 Docker Compose Configuration

```yaml
version: "3.8"

services:
  # Object Storage
  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - ./data/minio:/data
    environment:
      MINIO_ROOT_USER: ${MINIO_ROOT_USER:-minioadmin}
      MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD:-minioadmin}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Vector Database
  chroma:
    image: chromadb/chroma:latest
    ports:
      - "8000:8000"
    volumes:
      - ./data/chroma:/chroma/chroma
    environment:
      ANONYMIZED_TELEMETRY: "false"

  # LLM Server
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ./models:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # API Gateway
  api-gateway:
    build: ./services/api-gateway
    ports:
      - "8080:8000"
    environment:
      - INGESTION_URL=http://ingestion:8000
      - RETRIEVAL_URL=http://retrieval:8000
      - QUERY_URL=http://query:8000
    depends_on:
      - ingestion
      - retrieval
      - query

  # Ingestion Service
  ingestion:
    build: ./services/ingestion
    volumes:
      - ./data/documents:/app/documents
    environment:
      - MINIO_ENDPOINT=minio:9000
      - CHROMA_HOST=http://chroma:8000
    depends_on:
      - minio
      - chroma

  # Retrieval Service
  retrieval:
    build: ./services/retrieval
    environment:
      - CHROMA_HOST=http://chroma:8000
      - MINIO_ENDPOINT=minio:9000
    depends_on:
      - chroma
      - minio

  # Query Service
  query:
    build: ./services/query
    environment:
      - OLLAMA_HOST=http://ollama:11434
      - LLM_MODEL=llama3.1:8b
    depends_on:
      - ollama

volumes:
  minio_data:
  chroma_data:
  ollama_models:
```

### 5.2 Development Workflow

```bash
# Start all services
docker compose up -d

# Pull LLM model (first time)
docker compose exec ollama ollama pull llama3.1:8b

# View logs
docker compose logs -f api-gateway

# Rebuild after code changes
docker compose build ingestion
docker compose up -d ingestion

# Stop all services
docker compose down

# Full reset (delete data)
docker compose down -v
```

### 5.3 Hardware Requirements

| Configuration | CPU | RAM | GPU | Storage | Suitable For |
|---------------|-----|-----|-----|---------|--------------|
| Minimal | 4 cores | 16GB | None | 50GB SSD | Small models, testing |
| Recommended | 8 cores | 32GB | 8GB VRAM | 100GB SSD | 7-8B models |
| Performance | 16 cores | 64GB | 24GB VRAM | 500GB NVMe | 70B models |

---

## 6. Cost Characteristics

### 6.1 Hardware Costs (One-Time)

| Component | Minimal | Recommended | Performance |
|-----------|---------|-------------|-------------|
| CPU | Existing | Existing | $0 |
| RAM | +16GB ~$50 | +32GB ~$100 | +64GB ~$200 |
| GPU | None | RTX 4060 ~$300 | RTX 4090 ~$1,600 |
| Storage | Existing | +256GB SSD ~$30 | +1TB NVMe ~$100 |
| **Total** | **~$50** | **~$430** | **~$1,900** |

### 6.2 Operational Costs

| Component | Cost |
|-----------|------|
| Electricity | ~$5-20/month (GPU running) |
| Internet | Existing |
| Cloud services | $0 |
| **Total Monthly** | **$5-20** |

### 6.3 Comparison to Cloud

| Scenario | Local | Cloud Equivalent |
|----------|-------|------------------|
| 1K queries/day (1 month) | $10 electricity | $500-800 (GCP) |
| 10K queries/day (1 month) | $20 electricity | $2,000-4,000 (GCP) |
| Hardware amortized (2 years) | $18-80/month | N/A |

---

## 7. Operational Trade-offs

### 7.1 Advantages

- **Near-Zero Recurring Cost:** Only electricity
- **Full Control:** Complete access to every component
- **Privacy:** No data leaves local environment
- **Learning:** Deep understanding of full stack
- **No Quotas:** Unlimited requests (hardware-limited)
- **Offline Capable:** Works without internet

### 7.2 Disadvantages

- **No Automatic Scaling:** Fixed hardware capacity
- **Operational Burden:** You manage everything
- **Hardware Investment:** Upfront GPU cost
- **No HA/DR:** Single point of failure
- **Limited Enterprise Credibility:** Requires explanation

---

## 8. Migration Paths

### 8.1 To GCP Hybrid Architecture

| Local Component | GCP Hybrid Equivalent | Migration Steps |
|-----------------|----------------------|-----------------|
| MinIO | Cloud Storage | Use `mc mirror` or `gsutil` |
| ChromaDB | FAISS on Cloud Run | Export/import embeddings |
| Ollama | vLLM on Cloud Run | Deploy container |
| Docker Compose | Cloud Run services | Containerize and deploy |

**Migration Checklist:**
1. Create GCP project and enable APIs
2. Set up Cloud Run services
3. Deploy MinIO → GCS migration
4. Deploy embedding service to Cloud Run
5. Deploy vector search to Cloud Run
6. Deploy LLM container (vLLM) to Cloud Run
7. Update API gateway endpoints
8. Test end-to-end
9. Cut over DNS

### 8.2 To Full GCP-Native Architecture

| Local Component | GCP Native | Migration Steps |
|-----------------|------------|-----------------|
| MinIO | Cloud Storage | Direct upload |
| sentence-transformers | Vertex AI Embeddings | Re-embed all documents |
| ChromaDB | Vertex AI Vector Search | Create new index |
| Ollama | Vertex AI Gemini | Update API calls |

**Effort Estimate:** 2-4 weeks, primarily re-embedding

### 8.3 From GCP Variants

**Why migrate to local:**
- Cost reduction for low-traffic scenarios
- Development environment parity
- Privacy requirements
- Offline capability needs

**Steps:**
1. Export documents from Cloud Storage
2. Export embedding vectors (if compatible dimensions)
3. Set up local Docker Compose
4. Import data to MinIO
5. Import vectors to ChromaDB (or re-embed)
6. Deploy and test locally

---

## 9. Constraints and Non-Goals

### 9.1 Constraints

**Hardware Constraints:**
- Requires local hardware with sufficient GPU for acceptable performance
- Model quality directly limited by available VRAM
- Single machine; no distributed computing
- Network latency limited to LAN

**Operational Constraints:**
- No automatic scaling beyond hardware capacity
- Manual model updates and maintenance
- Single point of failure (no HA/DR)
- You are responsible for all infrastructure

**Capability Constraints:**
- Maximum concurrent users limited by hardware
- Model size limited by GPU memory
- Throughput limited by GPU compute

### 9.2 Non-Goals

**Explicitly Out of Scope:**

| Non-Goal | Rationale |
|----------|----------|
| Multi-user concurrent access at scale | Hardware-limited; use cloud variants |
| High availability / disaster recovery | Single machine deployment |
| Regulatory compliance (SOC2, HIPAA) | No audit controls or certifications |
| Production SLAs | No formal uptime guarantees |
| Multi-region deployment | Local-only by design |
| Internet-facing deployment | Use for development, not public production |
| Enterprise security controls | Basic auth only; no SSO, audit logs |

**Scaling Expectations:**
- This architecture is designed for **single-user or small team** workloads
- Suitable for development, learning, and demonstration
- NOT designed for production traffic beyond local testing
- Scales vertically only (better GPU = better performance)

**Availability Expectations:**
- Depends entirely on local hardware reliability
- No redundancy or failover
- Expect occasional downtime for maintenance

**This variant is NOT production-grade. Use Hybrid or GCP-Native for production deployments.**

---

## 10. Implementation Guidance

### 10.1 Service Classification

| Service | Type | State | Scaling |
|---------|------|-------|---------|
| API Gateway | Long-running container | Stateless | Docker replicas |
| Ingestion Service | Long-running container | Stateless | Docker replicas |
| Retrieval Service | Long-running container | Stateless | Docker replicas |
| Query Service | Long-running container | Stateless | Docker replicas |
| MinIO | Long-running container | Persistent | Single instance |
| ChromaDB | Long-running container | Persistent | Single instance |
| Ollama | Long-running container | Stateful (model) | Single instance |

### 10.2 Repository Structure

```
enterprise-rag-platform/
├── services/
│   ├── api-gateway/
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   ├── auth.py
│   │   └── requirements.txt
│   ├── ingestion/
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   ├── parser.py
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   └── requirements.txt
│   ├── retrieval/
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   ├── search.py
│   │   ├── ranker.py
│   │   └── requirements.txt
│   └── query/
│       ├── Dockerfile
│       ├── main.py
│       ├── context.py
│       ├── llm.py
│       └── requirements.txt
├── shared/
│   ├── models/
│   │   └── schemas.py
│   ├── utils/
│   │   └── config.py
│   └── clients/
│       ├── minio_client.py
│       ├── chroma_client.py
│       └── ollama_client.py
├── data/
│   ├── documents/     # Raw document storage
│   ├── minio/         # MinIO persistence
│   └── chroma/        # ChromaDB persistence
├── models/            # Downloaded LLM models
├── scripts/
│   ├── setup.sh
│   ├── pull_models.sh
│   └── seed_data.sh
├── docker-compose.yaml
├── docker-compose.dev.yaml
├── .env.example
├── Makefile
└── README.md
```

### 10.3 Incremental Implementation Order

1. **Phase 1: Infrastructure**
   - Set up Docker Compose with MinIO, ChromaDB, Ollama
   - Verify services start and communicate
   - Pull initial LLM model

2. **Phase 2: Ingestion Pipeline**
   - Implement document parser
   - Implement text chunker
   - Implement embedding generator
   - Store chunks and vectors

3. **Phase 3: Retrieval Pipeline**
   - Implement query embedding
   - Implement vector search
   - Implement result ranking

4. **Phase 4: Generation Pipeline**
   - Implement context packer
   - Implement prompt builder
   - Implement LLM caller
   - Implement response parser

5. **Phase 5: API Layer**
   - Implement API gateway
   - Add basic authentication
   - Add request validation

6. **Phase 6: Polish**
   - Add logging
   - Add health checks
   - Create CLI tools
   - Write documentation

---

## 11. Reference Configuration

### 11.1 Embedding Service

```python
# services/ingestion/embedder.py
from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True
        )
```

### 11.2 ChromaDB Integration

```python
# shared/clients/chroma_client.py
import chromadb
from chromadb.config import Settings

class ChromaClient:
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.client = chromadb.HttpClient(host=host, port=port)
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
    
    def upsert(self, ids: list[str], embeddings: list, metadatas: list[dict]):
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
        )
    
    def query(self, embedding: list, n_results: int = 5) -> dict:
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
```

### 11.3 Ollama Integration

```python
# shared/clients/ollama_client.py
import httpx
from typing import Optional

class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host
        self.client = httpx.Client(timeout=120.0)
    
    def generate(
        self,
        prompt: str,
        model: str = "llama3.1:8b",
        temperature: float = 0.2,
        max_tokens: int = 2048
    ) -> str:
        response = self.client.post(
            f"{self.host}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
        )
        return response.json()["response"]
```

### 11.4 Query Service Prompt Template

```python
# services/query/prompts.py
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based only on the provided context.

Rules:
1. Only use information from the context below
2. If the answer is not in the context, say "I cannot find this information in the provided documents"
3. Cite your sources using [Source: filename] format
4. Be concise and factual"""

def build_prompt(context: str, query: str) -> str:
    return f"""{SYSTEM_PROMPT}

Context:
{context}

Question: {query}

Answer:"""
```
