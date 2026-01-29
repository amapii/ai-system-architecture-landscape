# Enterprise Knowledge AI Platform: Hybrid Architecture

## 1. Positioning

**Deployment Mode:** Hybrid (GCP Infrastructure + Open-Source Models)

**Target Audience:**
- Freelancers and consultants
- Small-to-medium enterprises (SMEs)
- Startups with cost constraints
- Teams transitioning from local to cloud
- Production MVPs with growth potential

**Use Cases:**
- Production-ready RAG with cost control
- Client-facing demos with enterprise infrastructure
- Scalable prototypes with upgrade path
- Cost-optimized knowledge systems

---

## 2. System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENTS                                        │
│                     Web App / API Consumers / SDKs                          │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │ HTTPS
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CLOUD LOAD BALANCER                                  │
│                     (Cloud Run default endpoint)                            │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         API GATEWAY SERVICE                                 │
│                    Cloud Run (FastAPI + Auth)                               │
│              ┌──────────────────────────────────────────────┐               │
│              │  - Request validation                        │               │
│              │  - API key authentication                    │               │
│              │  - Rate limiting                             │               │
│              │  - Request routing                           │               │
│              └──────────────────────────────────────────────┘               │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────┐
        │                             │                         │
        ▼                             ▼                         ▼
┌───────────────┐    ┌────────────────────┐    ┌───────────────────────┐
│  INGESTION    │    │  RETRIEVAL SERVICE │    │  QUERY SERVICE        │
│  SERVICE      │    │  (Cloud Run)       │    │  (Cloud Run)          │
│  (Cloud Run)  │    │                    │    │                       │
│               │    │  - Query rewrite   │    │  - Context assembly   │
│  - Parser     │    │  - Vector search   │    │  - LLM invocation     │
│  - Chunker    │    │  - Hybrid filter   │    │  - Response format    │
│  - Embedder   │    │  - Ranking         │    │  - Citation attach    │
└───────┬───────┘    └─────────┬──────────┘    └───────────┬───────────┘
        │                      │                           │
        │                      │                           │
        ▼                      ▼                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HYBRID SERVICES LAYER                                    │
├───────────────────┬───────────────────┬─────────────────────────────────────┤
│ Cloud Storage     │ Vector Search     │ LLM Inference                       │
│ (GCS - Managed)   │ (Open-Source)     │ (Open Models)                       │
│                   │                   │                                     │
│ - Raw documents   │ FAISS or ChromaDB │ vLLM on Cloud Run / GKE            │
│ - Processed       │ on Cloud Run      │                                     │
│   chunks          │                   │ Llama 3.1 / Mistral / Gemma         │
└───────────────────┴───────────────────┴─────────────────────────────────────┘
        │                      │                           │
        │                      │                           │
        ▼                      ▼                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EMBEDDING SERVICE (Cloud Run)                            │
│              Open-Source Model (bge-base-en-v1.5 / nomic-embed-text)        │
└─────────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SECURITY & OBSERVABILITY                               │
├───────────────────────┬───────────────────────┬─────────────────────────────┤
│ Cloud IAM             │ Cloud Logging         │ Secret Manager              │
│ - Service accounts    │ - Structured logs     │ - API keys                  │
│ - Workload identity   │ - Log-based alerts    │ - Credentials               │
└───────────────────────┴───────────────────────┴─────────────────────────────┘
```

---

## 3. Component Architecture

### 3.1 Data Storage Layer

| Component | Service | Type | Specification |
|-----------|---------|------|---------------|
| Raw Document Store | Cloud Storage | Managed | Standard tier, regional |
| Processed Chunks | Cloud Storage | Managed | JSON format |
| Metadata Store | Firestore | Managed | Native mode (optional) |
| Vector Index | Cloud Storage | Managed | FAISS index files |

**State:** Persistent (managed)
**Deployment:** GCP-managed, no containers

**Cost Profile:**
- Cloud Storage: ~$0.02/GB/month
- Firestore: ~$0.18/100K reads

---

### 3.2 Embedding Service

| Component | Implementation | Specification |
|-----------|----------------|---------------|
| Embedding Model | Open-source (containerized) | bge-base-en-v1.5 (768 dim) |
| Runtime | Cloud Run | Scale-to-zero |

**Model Options:**
| Model | Dimensions | Container Size | Latency | Quality |
|-------|------------|----------------|---------|---------|
| all-MiniLM-L6-v2 | 384 | 300MB | 50ms | Good |
| bge-base-en-v1.5 | 768 | 800MB | 100ms | Better |
| nomic-embed-text | 768 | 1GB | 120ms | Better |

**Cloud Run Configuration:**
```yaml
spec:
  containerConcurrency: 4
  timeoutSeconds: 60
  resources:
    limits:
      cpu: "2"
      memory: "4Gi"
```

**State:** Stateless (model in container)
**Deployment:** Cloud Run (scale-to-zero)

**Implementation:**
```python
# services/embedding/main.py
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel

app = FastAPI()
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

class EmbedRequest(BaseModel):
    texts: list[str]

@app.post("/embed")
async def embed(request: EmbedRequest):
    embeddings = model.encode(
        request.texts,
        normalize_embeddings=True
    ).tolist()
    return {"embeddings": embeddings}
```

---

### 3.3 Vector Search

| Component | Implementation | Specification |
|-----------|----------------|---------------|
| Vector Store | FAISS on Cloud Run | In-memory index |
| Alternative | ChromaDB on Cloud Run | Persistent mode |
| Index Storage | Cloud Storage | FAISS index file backup |

**FAISS Service Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                    VECTOR SEARCH SERVICE                    │
│                       (Cloud Run)                           │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Startup:                                             │  │
│  │  1. Download index from GCS                           │  │
│  │  2. Load FAISS index into memory                      │  │
│  │  3. Ready to serve                                    │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Query:                                               │  │
│  │  1. Receive query vector                              │  │
│  │  2. FAISS similarity search                           │  │
│  │  3. Return top-k IDs + scores                         │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Cloud Run Configuration:**
```yaml
spec:
  containerConcurrency: 80
  timeoutSeconds: 60
  resources:
    limits:
      cpu: "2"
      memory: "8Gi"  # Index loaded in memory
```

**State:** Stateful (index in memory, persisted to GCS)
**Deployment:** Cloud Run with minimum instances = 1

**Implementation:**
```python
# services/vector-search/main.py
import faiss
import numpy as np
from fastapi import FastAPI
from google.cloud import storage
import os

app = FastAPI()
index = None
id_map = []

@app.on_event("startup")
async def load_index():
    global index, id_map
    client = storage.Client()
    bucket = client.bucket(os.environ["GCS_BUCKET"])
    
    # Download index file
    blob = bucket.blob("indexes/faiss.index")
    blob.download_to_filename("/tmp/faiss.index")
    index = faiss.read_index("/tmp/faiss.index")
    
    # Download ID map
    blob = bucket.blob("indexes/id_map.json")
    id_map = json.loads(blob.download_as_string())

@app.post("/search")
async def search(query_embedding: list[float], k: int = 5):
    query = np.array([query_embedding], dtype="float32")
    distances, indices = index.search(query, k)
    
    results = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < len(id_map):
            results.append({
                "id": id_map[idx],
                "score": float(dist),
                "rank": i + 1
            })
    return {"results": results}
```

---

### 3.4 LLM Inference

| Component | Implementation | Specification |
|-----------|----------------|---------------|
| LLM Server | vLLM on Cloud Run | OpenAI-compatible API |
| Alternative | vLLM on GKE | Higher throughput |
| Alternative | Ollama on GCE | GPU VM |

**Model Options:**
| Model | VRAM Required | Cloud Run Feasible | GKE Required |
|-------|---------------|-------------------|--------------|
| Llama-3.2-3B | 8GB | Yes (L4 GPU) | No |
| Llama-3.1-8B | 16GB | Yes (L4 GPU) | Optional |
| Mistral-7B | 16GB | Yes (L4 GPU) | Optional |
| Llama-3.1-70B | 80GB | No | Yes (A100) |

**Cloud Run with GPU Configuration:**
```yaml
# Cloud Run with L4 GPU
spec:
  containerConcurrency: 4
  timeoutSeconds: 300
  resources:
    limits:
      cpu: "8"
      memory: "32Gi"
      nvidia.com/gpu: "1"
```

**GKE Deployment (for larger models):**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
spec:
  replicas: 1
  template:
    spec:
      containers:
        - name: vllm
          image: vllm/vllm-openai:latest
          args:
            - "--model=meta-llama/Meta-Llama-3.1-8B-Instruct"
            - "--dtype=auto"
            - "--max-model-len=8192"
          resources:
            limits:
              nvidia.com/gpu: 1
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-l4
```

**State:** Stateless (model in container)
**Deployment:** Cloud Run (GPU) or GKE

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
- **Provider Substitution:** Swap open models for Vertex AI Gemini transparently
- **Cost Flexibility:** Route to different models based on complexity/cost
- **Migration Ready:** Same code works when upgrading to GCP-Native

**Implementation Requirement:**
- All application services MUST interact with LLMs exclusively via this API contract
- Direct SDK calls are prohibited in application code
- LLM client abstraction layer handles endpoint configuration

**Hybrid Implementation:**

vLLM provides native OpenAI-compatible API:

```python
# shared/clients/llm_client.py
import openai
import os

class VLLMClient:
    def __init__(self):
        self.client = openai.OpenAI(
            base_url=os.environ.get("LLM_SERVICE_URL", "http://localhost:8000/v1"),
            api_key="not-needed"  # vLLM doesn't require auth by default
        )
    
    def chat_completion(self, messages: list, model: str = None, **kwargs) -> dict:
        return self.client.chat.completions.create(
            model=model or os.environ.get("LLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
            messages=messages,
            **kwargs
        )
```

**Migration to GCP-Native:**

When upgrading to Vertex AI Gemini, only the client configuration changes:

```python
# Same interface, different backend
class GeminiClient:
    def chat_completion(self, messages: list, **kwargs) -> dict:
        # Wrap Gemini SDK to return OpenAI-compatible response
        ...
```

---

### 3.6 LLM Serving Model

**Architecture Pattern:** Long-Running Containerized Inference Server

**Key Characteristics:**
- LLMs are deployed as **persistent Cloud Run services or GKE pods**
- Models are loaded into GPU memory **once at container startup**
- Models are NOT loaded per request; this would be prohibitively slow
- Application services communicate via HTTP API calls to the inference endpoint

**Request Flow:**
```
Application Service
        │
        │ HTTPS POST /v1/chat/completions
        ▼
┌─────────────────────────┐
│ vLLM on Cloud Run       │
│ (or GKE)                │
│                         │
│ - Model in GPU memory   │
│ - L4/A100 GPU           │
│ - Persistent container  │
└─────────────────────────┘
```

**Conceptual Parity:**

This containerized inference pattern is functionally equivalent to:
- Vertex AI Gemini API (GCP-Native variant) - managed endpoint
- Ollama (Local variant) - local Docker container

The difference is deployment: containers on GCP with managed scaling.

**Service Characteristics:**
| Aspect | Specification |
|--------|--------------|
| Availability | Cloud Run SLA (99.95%) |
| Cold Start | 10-30 seconds (container + model loading) |
| Scaling | Auto-scaling with minimum instances |
| Model Loading | On container startup |
| Latency | ~200-800ms typical |

**Minimum Instances:**

To avoid cold starts, set minimum instances to 1:
```yaml
autoscaling.knative.dev/minScale: "1"
```

This incurs continuous cost but eliminates cold start latency.

---

### 3.7 Model Lifecycle

**Embedding Models:**
| Aspect | Hybrid Implementation |
|--------|----------------------|
| Weight Storage | Baked into container image |
| Loading | On container startup |
| Updates | Rebuild container, redeploy service |
| Version Control | Container image tag |

**LLM Models:**
| Aspect | Hybrid Implementation |
|--------|----------------------|
| Weight Storage | Baked into container or pulled from HuggingFace |
| Loading | On container startup (into GPU memory) |
| Updates | Rebuild container, redeploy service |
| Version Control | Model ID in container config |

**Model Update Process:**
1. Update model ID in Dockerfile or service configuration
2. Rebuild container: `docker build -t gcr.io/$PROJECT/llm:new-version .`
3. Push to Artifact Registry: `docker push gcr.io/$PROJECT/llm:new-version`
4. Redeploy Cloud Run service with new image

**Container Build Strategy:**

**Option 1: Bake model into image (faster startup, larger image)**
```dockerfile
FROM vllm/vllm-openai:latest
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('meta-llama/Meta-Llama-3.1-8B-Instruct')"
CMD ["--model", "meta-llama/Meta-Llama-3.1-8B-Instruct"]
```

**Option 2: Pull model at startup (smaller image, slower cold start)**
```dockerfile
FROM vllm/vllm-openai:latest
CMD ["--model", "meta-llama/Meta-Llama-3.1-8B-Instruct"]
# Model downloaded on first startup
```

**Rollback:**
- Deploy previous container image tag
- No data migration required
- Redeploy takes ~1-2 minutes

---

### 3.8 Application Services

| Service | Runtime | Responsibility | Scaling |
|---------|---------|----------------|---------|
| API Gateway | Cloud Run | Auth, routing, rate limiting | 0-100 instances |
| Ingestion Service | Cloud Run | Document processing | 0-10 instances |
| Embedding Service | Cloud Run | Vector generation | 0-20 instances |
| Vector Search | Cloud Run | Similarity search | 1-10 instances |
| Retrieval Service | Cloud Run | Query orchestration | 0-20 instances |
| Query Service | Cloud Run | LLM orchestration | 0-20 instances |
| LLM Service | Cloud Run (GPU) or GKE | Model inference | 1-5 instances |

**State:** All stateless except Vector Search (warm start) and LLM (model loaded)

---

### 3.6 Security Layer

| Component | GCP Service | Configuration |
|-----------|-------------|---------------|
| Identity | Cloud IAM | Service accounts per service |
| Authentication | API Keys | Stored in Secret Manager |
| Network | Cloud Run (public) | Optional VPC connector |
| Secrets | Secret Manager | Environment injection |

**IAM Roles Required:**
- `roles/run.invoker` - Service-to-service calls
- `roles/storage.objectViewer` - GCS read
- `roles/storage.objectCreator` - GCS write
- `roles/secretmanager.secretAccessor` - Secrets access

---

### 3.7 Observability

| Component | GCP Service | Purpose |
|-----------|-------------|---------|
| Logging | Cloud Logging | Structured logs |
| Monitoring | Cloud Monitoring | Metrics, dashboards |
| Alerting | Cloud Monitoring | Based on log metrics |

**Key Metrics:**
- Request latency (p50, p95, p99)
- Cold start frequency
- LLM token throughput
- Cost per query
- Error rates

---

## 4. Data Flow

### 4.1 Document Ingestion Flow

```
Document Upload
      │
      ▼
┌─────────────────┐
│ Ingestion       │
│ Service         │
│ (Cloud Run)     │
│                 │
│ 1. Parse doc    │
│ 2. Chunk text   │
│ 3. Clean chunks │
└────────┬────────┘
         │
         ├──────────────────────────────────┐
         │                                  │
         ▼                                  ▼
┌─────────────────┐           ┌────────────────────────┐
│ Cloud Storage   │           │ Embedding Service      │
│                 │           │ (Cloud Run)            │
│ Store chunks    │           │                        │
│ as JSON         │           │ bge-base-en-v1.5       │
└─────────────────┘           └────────────┬───────────┘
                                           │
                                           ▼
                              ┌────────────────────────┐
                              │ Index Builder Job      │
                              │ (Cloud Run Job)        │
                              │                        │
                              │ Build FAISS index      │
                              │ Upload to GCS          │
                              └────────────┬───────────┘
                                           │
                                           ▼
                              ┌────────────────────────┐
                              │ Vector Search Service  │
                              │ (Cloud Run)            │
                              │                        │
                              │ Hot-reload index       │
                              └────────────────────────┘
```

### 4.2 Query Flow

```
User Query
      │
      ▼
┌─────────────────┐
│ API Gateway     │
│ (Cloud Run)     │
│                 │
│ Authenticate    │
│ Rate limit      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Retrieval       │
│ Service         │
│ (Cloud Run)     │
│                 │
│ 1. Embed query  │──────▶ Embedding Service (Cloud Run)
│ 2. Vector search│──────▶ Vector Search Service (Cloud Run)
│ 3. Fetch chunks │──────▶ Cloud Storage
│ 4. Rank results │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Query Service   │
│ (Cloud Run)     │
│                 │
│ 1. Pack context │
│ 2. Build prompt │
│ 3. Call LLM     │──────▶ vLLM Service (Cloud Run GPU / GKE)
│ 4. Parse output │
│ 5. Add citations│
└────────┬────────┘
         │
         ▼
    Response
```

---

## 5. Deployment Model

### 5.1 Infrastructure as Code

**Terraform Structure:**
```
terraform/
├── main.tf
├── variables.tf
├── outputs.tf
├── modules/
│   ├── storage/
│   ├── cloud-run/
│   ├── iam/
│   └── secrets/
└── environments/
    ├── dev.tfvars
    └── prod.tfvars
```

### 5.2 CI/CD Pipeline

**Cloud Build Configuration:**
```yaml
steps:
  # Build embedding service
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/embedding:$COMMIT_SHA', './services/embedding']

  # Build vector search service
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/vector-search:$COMMIT_SHA', './services/vector-search']

  # Build LLM service
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/llm:$COMMIT_SHA', './services/llm']

  # Push all images
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', '--all-tags', 'gcr.io/$PROJECT_ID/embedding']

  # Deploy to Cloud Run
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        gcloud run deploy embedding --image=gcr.io/$PROJECT_ID/embedding:$COMMIT_SHA --region=us-central1
        gcloud run deploy vector-search --image=gcr.io/$PROJECT_ID/vector-search:$COMMIT_SHA --region=us-central1
```

### 5.3 Local Development

**Docker Compose for Local Development:**
```yaml
version: "3.8"

services:
  embedding:
    build: ./services/embedding
    ports:
      - "8001:8000"

  vector-search:
    build: ./services/vector-search
    ports:
      - "8002:8000"
    volumes:
      - ./data/indexes:/app/indexes

  llm:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ./models:/root/.ollama
```

---

## 6. Cost Characteristics

### 6.1 Cost Components

| Component | Pricing Model | Estimated Monthly Cost |
|-----------|---------------|------------------------|
| Cloud Run (CPU services) | CPU/memory/requests | $20-100 |
| Cloud Run (GPU LLM) | GPU-hours | $50-500 |
| Cloud Storage | Storage + egress | $5-20 |
| Secret Manager | Secrets + access | $1-5 |
| Cloud Logging | Ingestion | $10-50 |
| Artifact Registry | Storage | $5-10 |

### 6.2 Cost Optimization Strategies

- **Scale-to-zero:** All CPU services scale to zero when idle
- **Minimum instances:** Only Vector Search and LLM need min instances
- **Spot VMs (GKE):** Use preemptible nodes for LLM if fault-tolerant
- **Right-size containers:** Minimize memory allocation
- **Batch operations:** Aggregate embedding requests

### 6.3 Total Cost of Ownership

| Scale | Monthly Estimate | Breakdown |
|-------|------------------|-----------|
| MVP (1K queries/day) | $100-250 | LLM dominant |
| Growth (10K queries/day) | $400-800 | Balanced |
| Scale (50K queries/day) | $1,500-3,000 | Need GKE for LLM |

### 6.4 Comparison to Other Variants

| Variant | 1K queries/day | 10K queries/day |
|---------|----------------|-----------------|
| Full GCP-Native | $500-800 | $2,000-4,000 |
| **Hybrid** | **$100-250** | **$400-800** |
| Local Parity | $10-20 | $20-40 |

**Hybrid is 3-5x cheaper than Full GCP-Native at equivalent scale.**

---

## 7. Operational Trade-offs

### 7.1 Advantages

- **Low Idle Cost:** Scale-to-zero for most services
- **Model Flexibility:** Use any open-source model
- **GCP Infrastructure:** IAM, logging, networking built-in
- **Clear Upgrade Path:** Easy migration to full GCP-Native
- **No Vendor Lock-in on Models:** Switch LLMs without code changes
- **Production Ready:** Enterprise-grade infrastructure

### 7.2 Disadvantages

- **Cold Starts:** Vector search and LLM need warm instances
- **GPU Availability:** Cloud Run GPU regions limited
- **Ops Overhead:** More containers to manage than full managed
- **Index Rebuild:** Manual process for large updates
- **Model Updates:** Container rebuilds for model changes

---

## 8. Migration Paths

### 8.1 From Local Parity Architecture

| Local Component | Hybrid Equivalent | Migration Steps |
|-----------------|-------------------|-----------------|
| MinIO | Cloud Storage | gsutil upload |
| ChromaDB | FAISS on Cloud Run | Export/re-index |
| sentence-transformers | Embedding Service (Cloud Run) | Same model, containerized |
| Ollama | vLLM on Cloud Run | Model conversion if needed |
| Docker Compose | Cloud Run | Deploy containers |

**Migration Checklist:**
1. Create GCP project and enable APIs
2. Upload documents to Cloud Storage
3. Deploy Embedding Service
4. Re-embed all documents
5. Build FAISS index, upload to GCS
6. Deploy Vector Search Service
7. Deploy LLM Service (vLLM)
8. Deploy API Gateway and orchestration services
9. Test end-to-end
10. Cut over

**Effort Estimate:** 1-2 weeks

### 8.2 To Full GCP-Native Architecture

| Hybrid Component | GCP Native | Migration Steps |
|------------------|------------|-----------------|
| Open embedding model | Vertex AI Embeddings | Re-embed all docs (dimensions may change) |
| FAISS on Cloud Run | Vertex AI Vector Search | Create managed index |
| vLLM on Cloud Run | Vertex AI Gemini | Update API calls |

**Why Migrate:**
- Need managed scaling beyond Cloud Run limits
- Require enterprise SLAs
- Want to eliminate container ops

**Migration Checklist:**
1. Enable Vertex AI APIs
2. Create Vector Search index
3. Re-embed with Vertex AI Embeddings
4. Index in Vector Search
5. Update Query Service for Gemini API
6. Test and validate quality
7. Cut over

**Effort Estimate:** 2-4 weeks (primarily re-embedding)

### 8.3 From Full GCP-Native Architecture

**Why Migrate:**
- Reduce costs
- Need custom model fine-tuning
- Avoid vendor lock-in

| GCP Native | Hybrid Equivalent | Steps |
|------------|-------------------|-------|
| Vertex AI Embeddings | Open embedding model | Re-embed (dimension match) |
| Vertex AI Vector Search | FAISS on Cloud Run | Export/rebuild index |
| Vertex AI Gemini | vLLM on Cloud Run | Update API endpoints |

**Effort Estimate:** 2-3 weeks

---

## 9. Constraints and Non-Goals

### 9.1 Constraints

**Infrastructure Constraints:**
- Cloud Run GPU availability limited to specific regions (us-central1, europe-west4)
- FAISS index size limited by Cloud Run memory (max 32GB)
- Model size limited by GPU memory (L4: 24GB, A100: 40GB/80GB)

**Operational Constraints:**
- Cold start latency for LLM service (10-30 seconds without minimum instances)
- Container rebuilds required for model updates
- Index rebuilds required for large document updates
- More ops overhead than fully managed GCP-Native

**Cost Constraints:**
- GPU instances must run continuously for low latency (minimum instances cost)
- Index service requires warm instances
- Container image storage costs

### 9.2 Non-Goals

**Explicitly Out of Scope:**

| Non-Goal | Rationale |
|----------|----------|
| Multi-region deployment | Single region target for simplicity |
| Real-time streaming | Batch responses only |
| Custom model fine-tuning | Use base models; fine-tuning is a separate workflow |
| Agent-based architectures | Deterministic orchestration only |
| Sub-100ms latency | Cold starts and inference time prevent this |
| 70B+ models on Cloud Run | GPU memory limits; use GKE for larger models |
| Zero-downtime deployments | Container restarts cause brief interruptions |

**Scaling Expectations:**
- This architecture is designed for **production MVP to mid-scale** workloads
- Suitable for 1K-50K queries/day
- Beyond 50K/day, consider GKE for LLM or migrate to GCP-Native
- Horizontal scaling limited by GPU availability

**Availability Expectations:**
- Cloud Run: 99.95% uptime SLA
- With minimum instances: near-zero cold starts
- Without minimum instances: expect 10-30s cold starts

**This variant is production-ready for small-to-medium scale deployments.**

---

## 10. Implementation Guidance

### 10.1 Service Classification

| Service | Type | State | Scaling | Min Instances |
|---------|------|-------|---------|---------------|
| API Gateway | Long-running | Stateless | 0-100 | 0 |
| Ingestion Service | Long-running | Stateless | 0-10 | 0 |
| Embedding Service | Long-running | Stateless | 0-20 | 0 |
| Vector Search | Long-running | Stateful* | 1-10 | 1 |
| Retrieval Service | Long-running | Stateless | 0-20 | 0 |
| Query Service | Long-running | Stateless | 0-20 | 0 |
| LLM Service | Long-running | Stateful* | 1-5 | 1 |
| Index Builder | Job | Stateless | N/A | N/A |

*Stateful = model/index loaded in memory, requires warm instance

### 10.2 Repository Structure

```
enterprise-rag-platform/
├── services/
│   ├── api-gateway/
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   └── requirements.txt
│   ├── ingestion/
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   ├── parser.py
│   │   └── chunker.py
│   ├── embedding/
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   └── requirements.txt
│   ├── vector-search/
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   └── requirements.txt
│   ├── retrieval/
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   └── requirements.txt
│   ├── query/
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   └── requirements.txt
│   └── llm/
│       ├── Dockerfile
│       └── requirements.txt
├── jobs/
│   └── index-builder/
│       ├── Dockerfile
│       ├── main.py
│       └── requirements.txt
├── shared/
│   ├── models/
│   │   └── schemas.py
│   ├── clients/
│   │   ├── gcs_client.py
│   │   ├── embedding_client.py
│   │   └── llm_client.py
│   └── config/
│       └── settings.py
├── terraform/
│   ├── main.tf
│   ├── cloud-run.tf
│   ├── storage.tf
│   └── iam.tf
├── cloudbuild.yaml
├── docker-compose.yaml  # Local development
├── Makefile
└── README.md
```

### 10.3 Incremental Implementation Order

1. **Phase 1: Foundation**
   - Create GCP project, enable APIs
   - Set up Cloud Storage buckets
   - Configure IAM service accounts
   - Deploy Secret Manager secrets

2. **Phase 2: Embedding Pipeline**
   - Implement Embedding Service
   - Deploy to Cloud Run
   - Test embedding generation

3. **Phase 3: Ingestion Pipeline**
   - Implement document parser
   - Implement chunker
   - Deploy Ingestion Service
   - Test document upload flow

4. **Phase 4: Vector Search**
   - Implement Index Builder job
   - Build initial FAISS index
   - Deploy Vector Search Service
   - Test similarity search

5. **Phase 5: LLM Service**
   - Build vLLM container with model
   - Deploy to Cloud Run (GPU) or GKE
   - Test inference endpoint

6. **Phase 6: Orchestration**
   - Implement Retrieval Service
   - Implement Query Service
   - Deploy and test end-to-end

7. **Phase 7: API Layer**
   - Implement API Gateway
   - Add authentication
   - Deploy and test

8. **Phase 8: Observability**
   - Configure Cloud Logging
   - Set up dashboards
   - Add alerting

---

## 11. Reference Configuration

### 11.1 Embedding Service Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    sentence-transformers \
    torch --index-url https://download.pytorch.org/whl/cpu

COPY main.py .

# Pre-download model during build
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### 11.2 vLLM Service Dockerfile

```dockerfile
FROM vllm/vllm-openai:latest

# Pre-download model during build (optional, can also mount)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('meta-llama/Meta-Llama-3.1-8B-Instruct')"

CMD ["--model", "meta-llama/Meta-Llama-3.1-8B-Instruct", "--dtype", "auto", "--max-model-len", "8192"]
```

### 11.3 Cloud Run Service Definition

```yaml
# cloud-run-service.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: retrieval-service
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "0"
        autoscaling.knative.dev/maxScale: "20"
        run.googleapis.com/execution-environment: gen2
    spec:
      containerConcurrency: 80
      timeoutSeconds: 60
      serviceAccountName: retrieval-service@PROJECT_ID.iam.gserviceaccount.com
      containers:
        - image: gcr.io/PROJECT_ID/retrieval-service:latest
          ports:
            - containerPort: 8080
          resources:
            limits:
              cpu: "2"
              memory: "2Gi"
          env:
            - name: EMBEDDING_SERVICE_URL
              value: https://embedding-service-xxx.run.app
            - name: VECTOR_SEARCH_URL
              value: https://vector-search-xxx.run.app
            - name: GCS_BUCKET
              value: PROJECT_ID-documents
```

### 11.4 Query Service Implementation

```python
# services/query/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import os

app = FastAPI()

LLM_URL = os.environ["LLM_SERVICE_URL"]

class QueryRequest(BaseModel):
    query: str
    context: list[str]
    max_tokens: int = 1024

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

SYSTEM_PROMPT = """You are a helpful assistant answering questions based on provided context.
Rules:
1. Only use information from the context
2. Cite sources using [Source: X] format
3. If information is not in context, say so
4. Be concise and factual"""

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    context_text = "\n\n".join([
        f"[Source {i+1}]: {chunk}"
        for i, chunk in enumerate(request.context)
    ])
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{LLM_URL}/v1/chat/completions",
            json={
                "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {request.query}"}
                ],
                "temperature": 0.2,
                "max_tokens": request.max_tokens
            },
            timeout=120.0
        )
    
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail="LLM service error")
    
    result = response.json()
    answer = result["choices"][0]["message"]["content"]
    
    return QueryResponse(
        answer=answer,
        sources=[f"Source {i+1}" for i in range(len(request.context))]
    )
```

---

## 12. Upgrade Path to Full GCP-Native

When traffic or requirements exceed Hybrid capabilities:

### 12.1 Trigger Conditions

- FAISS index exceeds 32GB (Cloud Run memory limit)
- Need for automatic index updates (Vector Search streaming)
- Require multi-region deployment
- Enterprise SLA requirements
- Team prefers managed services over container ops

### 12.2 Migration Steps

| Step | Action | Effort |
|------|--------|--------|
| 1 | Enable Vertex AI APIs | 1 hour |
| 2 | Create Vector Search index | 2 hours |
| 3 | Re-embed with Vertex AI Embeddings | 1-2 days |
| 4 | Index embeddings in Vector Search | 4-8 hours |
| 5 | Update Retrieval Service for Vector Search | 1 day |
| 6 | Update Query Service for Gemini | 1 day |
| 7 | Test and validate | 2-3 days |
| 8 | Gradual traffic migration | 1-2 days |

### 12.3 Code Changes Required

```python
# Before (Hybrid - open model)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-base-en-v1.5")
embeddings = model.encode(texts)

# After (GCP-Native - Vertex AI)
from vertexai.language_models import TextEmbeddingModel
model = TextEmbeddingModel.from_pretrained("text-embedding-004")
embeddings = model.get_embeddings(texts)
```

```python
# Before (Hybrid - vLLM)
client = openai.OpenAI(base_url="https://vllm-service.run.app/v1")
response = client.chat.completions.create(model="llama-3.1-8b", ...)

# After (GCP-Native - Gemini)
from vertexai.generative_models import GenerativeModel
model = GenerativeModel("gemini-1.5-flash")
response = model.generate_content(prompt)
```
