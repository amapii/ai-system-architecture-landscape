# Enterprise Knowledge AI Platform: Full GCP-Native Architecture

## 1. Positioning

**Deployment Mode:** Full GCP-Native (Managed, Enterprise-Grade)

**Target Audience:**
- Enterprises with existing GCP investment
- Regulated industries (finance, healthcare, legal)
- Organizations prioritizing reliability over cost optimization
- Teams with limited DevOps capacity
- Production systems requiring SLAs

**Use Cases:**
- Enterprise knowledge base with audit requirements
- Compliance-sensitive document Q&A
- High-availability internal support systems
- Multi-region deployment requirements

---

## 2. System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENTS                                        │
│                     Web App / API Consumers / SDKs                          │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │ HTTPS
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CLOUD LOAD BALANCER                                  │
│                     (Global HTTP(S) LB + Cloud Armor)                       │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         API GATEWAY SERVICE                                 │
│                    Cloud Run (FastAPI + Auth Middleware)                    │
│              ┌──────────────────────────────────────────────┐               │
│              │  - Request validation                        │               │
│              │  - Rate limiting                             │               │
│              │  - Session management                        │               │
│              │  - Request routing                           │               │
│              └──────────────────────────────────────────────┘               │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
        ▼                         ▼                         ▼
┌───────────────┐    ┌────────────────────┐    ┌───────────────────────┐
│  INGESTION    │    │  RETRIEVAL SERVICE │    │  QUERY SERVICE        │
│  SERVICE      │    │  (Cloud Run)       │    │  (Cloud Run)          │
│  (Cloud Run)  │    │                    │    │                       │
│               │    │  - Query rewrite   │    │  - Context assembly   │
│  - Document   │    │  - Vector search   │    │  - LLM invocation     │
│    parsing    │    │  - Hybrid filter   │    │  - Response format    │
│  - Chunking   │    │  - Ranking         │    │  - Citation attach    │
│  - Embedding  │    │                    │    │                       │
└───────┬───────┘    └─────────┬──────────┘    └───────────┬───────────┘
        │                      │                           │
        │                      │                           │
        ▼                      ▼                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MANAGED SERVICES LAYER                            │
├───────────────────┬───────────────────┬─────────────────────────────────────┤
│ Cloud Storage     │ Vertex AI         │ Vertex AI                           │
│ (GCS)             │ Vector Search     │ Gemini API                          │
│                   │                   │                                     │
│ - Raw documents   │ - Embedding index │ - gemini-1.5-flash                  │
│ - Processed       │ - Similarity      │ - gemini-1.5-pro                    │
│   chunks          │   search          │ - Structured output                 │
└───────────────────┴───────────────────┴─────────────────────────────────────┘
        │                      │                           │
        └──────────────────────┼───────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         VERTEX AI EMBEDDINGS                                │
│                    text-embedding-004 / text-multilingual-embedding-002     │
└─────────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SECURITY & OBSERVABILITY                               │
├───────────────────────┬───────────────────────┬─────────────────────────────┤
│ Cloud IAM             │ Cloud Logging         │ Secret Manager              │
│ - Service accounts    │ - Structured logs     │ - API keys                  │
│ - Workload identity   │ - Log-based metrics   │ - Credentials               │
│                       │                       │                             │
│ VPC Service Controls  │ Cloud Monitoring      │ Cloud KMS                   │
│ - Data perimeter      │ - Dashboards          │ - CMEK encryption           │
│ - Access boundaries   │ - Alerting            │ - Key rotation              │
└───────────────────────┴───────────────────────┴─────────────────────────────┘
```

---

## 3. Component Architecture

### 3.1 Data Storage Layer

| Component | GCP Service | Specification |
|-----------|-------------|---------------|
| Raw Document Store | Cloud Storage | Standard tier, regional |
| Processed Chunks | Cloud Storage | JSON/Parquet format |
| Metadata Store | Firestore | Native mode, regional |
| Analytics Store | BigQuery | For usage analytics, optional |

**State:** Persistent
**Deployment:** Managed, no containers

**Data Flow:**
1. Documents uploaded to `gs://{project}-documents/raw/`
2. Processed chunks stored in `gs://{project}-documents/chunks/`
3. Metadata indexed in Firestore collection `document_metadata`

---

### 3.2 Embedding Service

| Component | GCP Service | Specification |
|-----------|-------------|---------------|
| Embedding Generation | Vertex AI Embeddings API | text-embedding-004 (768 dim) |
| Batch Processing | Vertex AI Pipelines | For bulk ingestion |

**Model Options:**
- `text-embedding-004`: English, 768 dimensions, 2048 token limit
- `text-multilingual-embedding-002`: Multilingual, 768 dimensions

**State:** Stateless
**Deployment:** API call (no container required)

**Characteristics:**
- Rate limit: 600 requests/minute (default quota)
- Latency: ~50-100ms per request
- Cost: $0.0001 per 1K characters

---

### 3.3 Vector Search

| Component | GCP Service | Specification |
|-----------|-------------|---------------|
| Index | Vertex AI Vector Search | Streaming updates enabled |
| Index Endpoint | Vertex AI Vector Search | Public or private endpoint |

**Configuration:**
- Dimensions: 768 (match embedding model)
- Distance measure: DOT_PRODUCT_DISTANCE
- Shard size: SHARD_SIZE_MEDIUM (up to 1M vectors)
- Algorithm: Tree-AH for approximate nearest neighbor

**State:** Persistent (managed index)
**Deployment:** Managed endpoint

**Operational Notes:**
- Index deployment takes 30-60 minutes
- Use streaming updates for real-time ingestion
- Minimum 2 replicas for high availability

---

### 3.4 LLM Inference

| Component | GCP Service | Specification |
|-----------|-------------|---------------|
| Generation | Vertex AI Gemini API | gemini-1.5-flash (default) |
| Advanced Tasks | Vertex AI Gemini API | gemini-1.5-pro (complex reasoning) |

**Model Selection:**
- `gemini-1.5-flash`: Fast, cost-effective, 1M token context
- `gemini-1.5-pro`: Higher quality, complex reasoning

**Configuration:**
```python
generation_config = {
    "temperature": 0.2,       # Low for factual responses
    "top_p": 0.8,
    "max_output_tokens": 2048,
    "response_mime_type": "application/json"  # Structured output
}
```

**State:** Stateless
**Deployment:** API call (no container required)

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
- **Model Switching:** Change from Gemini to Llama to Mistral without code changes
- **Provider Substitution:** Swap between managed APIs and self-hosted inference
- **Cost Flexibility:** Route to different models based on complexity/cost
- **Testing:** Use local models in development, managed in production

**Implementation Requirement:**
- All application services MUST interact with LLMs exclusively via this API contract
- Direct SDK calls to vendor-specific APIs are prohibited in application code
- LLM client abstraction layer handles provider-specific authentication

**GCP-Native Adaptation:**

For Vertex AI Gemini, a thin compatibility layer wraps the Gemini SDK:

```python
# shared/clients/llm_client.py
from abc import ABC, abstractmethod

class LLMClient(ABC):
    @abstractmethod
    def chat_completion(self, messages: list, **kwargs) -> dict:
        pass

class GeminiClient(LLMClient):
    def __init__(self):
        from vertexai.generative_models import GenerativeModel
        self.model = GenerativeModel("gemini-1.5-flash")
    
    def chat_completion(self, messages: list, **kwargs) -> dict:
        # Convert OpenAI message format to Gemini format
        prompt = self._convert_messages(messages)
        response = self.model.generate_content(prompt, **kwargs)
        # Return OpenAI-compatible response structure
        return {
            "choices": [{"message": {"content": response.text}}]
        }
```

---

### 3.6 LLM Serving Model

**Architecture Pattern:** Managed API Endpoint

**Key Characteristics:**
- LLM inference is provided as a **managed, always-available API service**
- Models are NOT loaded per request; Google manages model lifecycle
- Application services communicate via HTTPS API calls
- No container or infrastructure management required

**Request Flow:**
```
Application Service
        │
        │ HTTPS POST /v1/chat/completions
        ▼
┌─────────────────────────┐
│ Vertex AI Gemini API    │
│ (Google-managed)        │
│                         │
│ - Model always loaded   │
│ - Auto-scaling          │
│ - Multi-region          │
└─────────────────────────┘
```

**Conceptual Parity:**

This managed API pattern is functionally equivalent to:
- Self-hosted vLLM server (Hybrid/Local variants)
- Ollama inference endpoint (Local variant)

The difference is operational: Google manages infrastructure, scaling, and model updates.

**Service Characteristics:**
| Aspect | Specification |
|--------|--------------|
| Availability | 99.9% SLA |
| Cold Start | None (always warm) |
| Scaling | Automatic, managed |
| Model Loading | Pre-loaded by Google |
| Latency | ~200-500ms typical |

---

### 3.7 Model Lifecycle

**Embedding Models:**
| Aspect | GCP-Native Implementation |
|--------|---------------------------|
| Weight Storage | Google-managed (not accessible) |
| Loading | Always loaded, managed by Google |
| Updates | Automatic via model versioning |
| Version Control | Explicit model ID (e.g., `text-embedding-004`) |

**LLM Models:**
| Aspect | GCP-Native Implementation |
|--------|---------------------------|
| Weight Storage | Google-managed (not accessible) |
| Loading | Always loaded, managed by Google |
| Updates | New model versions released by Google |
| Version Control | Explicit model ID (e.g., `gemini-1.5-flash-001`) |

**Model Update Process:**
1. Google releases new model version
2. Update model ID in application configuration
3. Redeploy affected services
4. No re-embedding required if dimensions unchanged

**Version Pinning:**
```python
# Pin to specific version for reproducibility
model = GenerativeModel("gemini-1.5-flash-001")

# Use latest stable version
model = GenerativeModel("gemini-1.5-flash")
```

**Rollback:**
- Revert to previous model version ID
- No data migration required
- Re-embedding required only if embedding model changes

---

### 3.8 Application Services

| Service | Runtime | Responsibility | Scaling |
|---------|---------|----------------|---------|
| API Gateway | Cloud Run | Auth, routing, rate limiting | 0-100 instances |
| Ingestion Service | Cloud Run | Document processing, chunking, embedding | 0-20 instances |
| Retrieval Service | Cloud Run | Query rewrite, vector search, ranking | 0-50 instances |
| Query Service | Cloud Run | Context assembly, LLM call, response formatting | 0-50 instances |

**Cloud Run Configuration:**
```yaml
# Example service configuration
spec:
  containerConcurrency: 80
  timeoutSeconds: 300
  resources:
    limits:
      cpu: "2"
      memory: "4Gi"
```

**State:** Stateless (all state in managed services)
**Deployment:** Container images in Artifact Registry

---

### 3.6 Security Layer

| Component | GCP Service | Configuration |
|-----------|-------------|---------------|
| Identity | Cloud IAM | Service accounts per service |
| Network | VPC + VPC-SC | Private service access |
| Secrets | Secret Manager | API keys, credentials |
| Encryption | Cloud KMS | CMEK for all storage |
| WAF | Cloud Armor | DDoS protection, geo-filtering |

**IAM Roles Required:**
- `roles/aiplatform.user` - Vertex AI access
- `roles/storage.objectViewer` - GCS read
- `roles/storage.objectCreator` - GCS write
- `roles/datastore.user` - Firestore access
- `roles/secretmanager.secretAccessor` - Secrets access

---

### 3.7 Observability

| Component | GCP Service | Purpose |
|-----------|-------------|---------|
| Logging | Cloud Logging | Structured logs, audit trail |
| Monitoring | Cloud Monitoring | Metrics, dashboards |
| Tracing | Cloud Trace | Request tracing |
| Profiling | Cloud Profiler | Performance analysis |

**Key Metrics to Track:**
- Request latency (p50, p95, p99)
- Vector search latency
- LLM response time
- Token consumption
- Error rates by service
- Cost per query

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
│                 │
│ 1. Parse doc    │
│ 2. Extract text │
│ 3. Chunk text   │
│ 4. Clean chunks │
└────────┬────────┘
         │
         ├──────────────────────────────────┐
         │                                  │
         ▼                                  ▼
┌─────────────────┐                ┌─────────────────┐
│ Cloud Storage   │                │ Vertex AI       │
│                 │                │ Embeddings      │
│ Store chunks    │                │                 │
│ as JSON         │                │ Generate        │
└─────────────────┘                │ embeddings      │
                                   └────────┬────────┘
                                            │
                                            ▼
                                   ┌─────────────────┐
                                   │ Vertex AI       │
                                   │ Vector Search   │
                                   │                 │
                                   │ Upsert vectors  │
                                   └─────────────────┘
```

### 4.2 Query Flow

```
User Query
      │
      ▼
┌─────────────────┐
│ API Gateway     │
│                 │
│ Authenticate    │
│ Rate limit      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Retrieval       │
│ Service         │
│                 │
│ 1. Query embed  │──────▶ Vertex AI Embeddings
│ 2. Vector search│──────▶ Vertex AI Vector Search
│ 3. Fetch chunks │──────▶ Cloud Storage
│ 4. Rank results │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Query Service   │
│                 │
│ 1. Pack context │
│ 2. Build prompt │
│ 3. Call LLM     │──────▶ Vertex AI Gemini
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
│   ├── networking/
│   ├── iam/
│   ├── vertex-ai/
│   └── cloud-run/
└── environments/
    ├── dev.tfvars
    ├── staging.tfvars
    └── prod.tfvars
```

### 5.2 CI/CD Pipeline

**Cloud Build Configuration:**
```yaml
steps:
  # Build container images
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/api-gateway:$COMMIT_SHA', './services/api-gateway']

  # Push to Artifact Registry
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/api-gateway:$COMMIT_SHA']

  # Deploy to Cloud Run
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: 'gcloud'
    args:
      - 'run'
      - 'deploy'
      - 'api-gateway'
      - '--image=gcr.io/$PROJECT_ID/api-gateway:$COMMIT_SHA'
      - '--region=us-central1'
```

### 5.3 Environment Promotion

| Environment | Purpose | Scaling | Data |
|-------------|---------|---------|------|
| dev | Development | Minimal replicas | Synthetic data |
| staging | Pre-production | Production-like | Anonymized data |
| prod | Production | Full scaling | Production data |

---

## 6. Cost Characteristics

### 6.1 Cost Components

| Component | Pricing Model | Estimated Monthly Cost |
|-----------|---------------|------------------------|
| Cloud Run | CPU/memory/requests | $50-500 |
| Vertex AI Embeddings | Per 1K characters | $10-100 |
| Vertex AI Vector Search | Index size + queries | $200-2000 |
| Vertex AI Gemini | Per 1K tokens | $100-1000 |
| Cloud Storage | Storage + egress | $10-50 |
| Cloud Logging | Ingestion + storage | $20-100 |

### 6.2 Cost Optimization Strategies

- Use `gemini-1.5-flash` as default, escalate to `pro` only when needed
- Implement aggressive caching at API Gateway
- Use batch embedding for bulk ingestion
- Set Cloud Run minimum instances to 0 for non-critical services
- Configure log retention based on compliance requirements

### 6.3 Total Cost of Ownership

| Scale | Monthly Estimate | Notes |
|-------|------------------|-------|
| MVP (1K queries/day) | $500-800 | Minimal index, single region |
| Growth (10K queries/day) | $2,000-4,000 | Vector Search dominant |
| Enterprise (100K queries/day) | $10,000-25,000 | Multi-region, HA |

---

## 7. Operational Trade-offs

### 7.1 Advantages

- **Minimal Ops Burden:** All core services are managed
- **Enterprise Security:** IAM, VPC-SC, CMEK out of the box
- **Scalability:** Automatic scaling, multi-region support
- **Audit Trail:** Built-in logging and compliance features
- **Support:** Google Cloud support and SLAs

### 7.2 Disadvantages

- **Vendor Lock-in:** Deep integration with GCP services
- **Higher Cost:** Premium for managed services
- **Less Control:** Limited customization of model behavior
- **Quota Management:** Need to request quota increases
- **Cold Starts:** Vector Search endpoint startup time

---

## 8. Migration Paths

### 8.1 From Local Parity Architecture

| Local Component | GCP Equivalent | Migration Steps |
|-----------------|----------------|-----------------|
| MinIO | Cloud Storage | Use `gsutil` for data transfer |
| FAISS | Vertex AI Vector Search | Re-index with managed service |
| vLLM | Vertex AI Gemini | Update API calls |
| Docker Compose | Cloud Run | Deploy containers |

**Migration Checklist:**
1. Provision GCP infrastructure (Terraform)
2. Migrate documents to Cloud Storage
3. Deploy Vector Search index
4. Re-embed documents with Vertex AI Embeddings
5. Index embeddings in Vector Search
6. Deploy services to Cloud Run
7. Update DNS/routing
8. Validate and cutover

### 8.2 From Hybrid Architecture

| Hybrid Component | GCP Native | Migration Steps |
|------------------|------------|-----------------|
| Open embedding model | Vertex AI Embeddings | Re-embed all documents |
| FAISS on Cloud Run | Vertex AI Vector Search | Create managed index |
| vLLM on Cloud Run | Vertex AI Gemini | Update API endpoints |

**Effort Estimate:** 2-4 weeks depending on data volume

### 8.3 To Hybrid Architecture

Primary reason: Cost reduction

**Reversible Steps:**
1. Replace Gemini with self-hosted open model
2. Replace Vector Search with FAISS on Cloud Run
3. Maintain GCP infrastructure (Cloud Run, GCS, IAM)

---

## 9. Constraints and Non-Goals

### 9.1 Constraints

**Infrastructure Constraints:**
- Requires GCP organization and billing account
- Vertex AI Vector Search minimum deployment cost (~$200/month)
- Gemini API quota limits may require pre-approval for high scale
- Data residency determined by region selection

**Operational Constraints:**
- Model behavior controlled by Google; no weight access
- Embedding dimensions fixed by model choice
- API rate limits apply per project
- Cold starts on Cloud Run services (not on Gemini API)

**Cost Constraints:**
- Per-token pricing for LLM inference
- Minimum monthly spend for Vector Search
- Egress costs for cross-region traffic

### 9.2 Non-Goals

**Explicitly Out of Scope:**

| Non-Goal | Rationale |
|----------|----------|
| Multi-cloud deployment | This variant is GCP-native by design |
| On-premise deployment | Use Local Parity variant for on-prem |
| Custom model fine-tuning | Use base models; fine-tuning requires separate workflow |
| Real-time streaming | Batch generation only; streaming adds complexity |
| Agent-based architectures | Deterministic orchestration preferred for auditability |
| Sub-100ms latency | Not achievable with managed LLM APIs |
| Offline operation | Requires internet connectivity to GCP |

**Scaling Expectations:**
- This architecture is designed for **enterprise-scale production** workloads
- Supports multi-region deployment with managed failover
- Automatic scaling to handle traffic spikes
- Enterprise SLAs available through Google Cloud support

**Availability Expectations:**
- Vertex AI: 99.9% uptime SLA
- Cloud Run: 99.95% uptime SLA
- Cloud Storage: 99.99% availability

**This variant is enterprise-grade and production-ready.**

---

## 10. Implementation Guidance

### 10.1 Service Classification

| Service | Type | State | Scaling |
|---------|------|-------|---------|
| API Gateway | Long-running service | Stateless | Horizontal |
| Ingestion Service | Long-running service | Stateless | Horizontal |
| Retrieval Service | Long-running service | Stateless | Horizontal |
| Query Service | Long-running service | Stateless | Horizontal |
| Batch Indexer | Scheduled job | Stateless | N/A |
| Index Builder | One-time job | Stateless | N/A |

### 10.2 Repository Structure

```
enterprise-rag-platform/
├── services/
│   ├── api-gateway/
│   │   ├── Dockerfile
│   │   ├── main.py
│   │   └── requirements.txt
│   ├── ingestion/
│   ├── retrieval/
│   └── query/
├── shared/
│   ├── models/
│   ├── utils/
│   └── config/
├── terraform/
│   ├── modules/
│   └── environments/
├── cloudbuild.yaml
├── docker-compose.yaml  # Local development
└── README.md
```

### 10.3 Incremental Implementation Order

1. **Phase 1: Foundation**
   - Set up GCP project and IAM
   - Deploy Cloud Storage buckets
   - Configure Firestore

2. **Phase 2: Ingestion**
   - Implement document parser
   - Deploy Ingestion Service
   - Integrate Vertex AI Embeddings

3. **Phase 3: Retrieval**
   - Deploy Vector Search index
   - Implement Retrieval Service
   - Add query rewriting

4. **Phase 4: Generation**
   - Implement Query Service
   - Integrate Gemini API
   - Add structured output parsing

5. **Phase 5: API Layer**
   - Deploy API Gateway
   - Add authentication
   - Implement rate limiting

6. **Phase 6: Observability**
   - Configure logging
   - Set up dashboards
   - Add alerting

---

## 11. Reference Configuration

### 11.1 Vertex AI Vector Search Index

```python
from google.cloud import aiplatform

aiplatform.init(project="PROJECT_ID", location="us-central1")

index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name="document-embeddings",
    contents_delta_uri="gs://bucket/embeddings/",
    dimensions=768,
    approximate_neighbors_count=150,
    distance_measure_type="DOT_PRODUCT_DISTANCE",
    leaf_node_embedding_count=500,
    leaf_nodes_to_search_percent=10,
)
```

### 11.2 Cloud Run Service

```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: retrieval-service
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "0"
        autoscaling.knative.dev/maxScale: "50"
    spec:
      containerConcurrency: 80
      timeoutSeconds: 300
      containers:
        - image: gcr.io/PROJECT_ID/retrieval-service:latest
          resources:
            limits:
              cpu: "2"
              memory: "4Gi"
          env:
            - name: PROJECT_ID
              value: "PROJECT_ID"
            - name: VECTOR_SEARCH_INDEX
              value: "projects/PROJECT_ID/locations/us-central1/indexes/INDEX_ID"
```

### 11.3 Gemini API Call

```python
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel

model = GenerativeModel("gemini-1.5-flash")

response = model.generate_content(
    f"""Based on the following context, answer the question.
    
Context:
{context}

Question: {query}

Provide a structured JSON response with:
- answer: The direct answer
- confidence: A score from 0-1
- citations: List of source references
""",
    generation_config={
        "temperature": 0.2,
        "response_mime_type": "application/json",
    }
)
```
