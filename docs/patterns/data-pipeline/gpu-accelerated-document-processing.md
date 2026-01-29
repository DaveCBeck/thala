---
name: gpu-accelerated-document-processing
title: "GPU-Accelerated Document Processing Service"
date: 2025-12-17
category: data-pipeline
applicability:
  - "High-volume PDF/DOCX/EPUB conversion requiring speed"
  - "Document processing with OCR and table extraction"
  - "RAG pipelines needing structured chunks from documents"
components: [pdf_processor, docker_service, async_task]
complexity: complex
verified_in_production: true
related_solutions: []
tags: [gpu, marker, celery, fastapi, docker, pdf, ocr, document-processing]
---

# GPU-Accelerated Document Processing Service

## Intent

Provide high-quality, GPU-accelerated document conversion (PDF, DOCX, EPUB to markdown) with async job processing, quality presets, and RAG-ready output chunks.

## Motivation

Document processing for RAG systems requires:
- High-quality text extraction preserving structure
- Table and figure recognition
- Multi-language OCR support
- Processing speed for batch operations
- Async job handling for long-running conversions

This pattern establishes a containerized service using Marker (GPU-accelerated document converter) with FastAPI for the API layer and Celery for async job processing, enabling scalable document processing without blocking the main application.

## Applicability

Use this pattern when:
- Processing many documents (PDFs, DOCX, EPUB) for RAG/search systems
- Needing high-quality OCR and table extraction
- GPU resources are available for acceleration
- Async processing is required (jobs can take seconds to minutes)

Do NOT use this pattern when:
- Only processing occasional single documents
- No GPU available (CPU-only will be very slow)
- Simple text extraction suffices (use PyPDF2 or similar)
- Real-time synchronous processing is required

## Structure

```
services/marker/
├── Dockerfile.gpu          # NVIDIA CUDA base image
├── docker-compose.yml      # Multi-container setup
├── requirements.txt        # Python dependencies
├── app/
│   ├── __init__.py
│   ├── config.py          # Settings + quality presets
│   ├── main.py            # FastAPI endpoints
│   ├── processor.py       # Marker wrapper with lazy loading
│   └── tasks.py           # Celery task definitions
└── data/
    ├── input/             # Documents to process
    └── output/            # Converted documents
```

**Service Architecture:**

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   FastAPI       │────▶│     Redis       │◀────│  Celery Worker  │
│   (API Layer)   │     │   (Job Queue)   │     │  (GPU Process)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                                               │
         │              ┌─────────────────┐              │
         └─────────────▶│     Flower      │◀─────────────┘
                        │  (Monitoring)   │
                        └─────────────────┘
```

## Implementation

### Step 1: Quality Presets Configuration

Define processing presets for different document types:

```python
# app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/0"
    google_api_key: str = ""  # For LLM mode (Gemini)

    # Batch sizes tuned for RTX 4060 Ti 16GB
    inference_ram: int = 16
    recognition_batch_size: int = 256
    detector_batch_size: int = 24
    layout_batch_size: int = 32
    table_rec_batch_size: int = 48

    class Config:
        env_file = ".env"


QUALITY_PRESETS = {
    "fast": {
        "batch_multiplier": 4,
        "force_ocr": False,
        "use_llm": False,
        "description": "Fast processing for good quality digital documents",
    },
    "balanced": {
        "batch_multiplier": 2,
        "force_ocr": False,
        "use_llm": False,
        "description": "Balanced quality/speed for general documents",
    },
    "quality": {
        "batch_multiplier": 1,
        "force_ocr": True,
        "use_llm": True,  # Uses Gemini for table/form extraction
        "description": "Highest quality for scanned/degraded documents",
    },
}
```

### Step 2: Document Processor with Lazy Model Loading

Wrap Marker with lazy model initialization (models are large and expensive to load):

```python
# app/processor.py
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.renderers.markdown import MarkdownRenderer
from marker.renderers.chunk import ChunkRenderer

class MarkerProcessor:
    """Wrapper around Marker PDF converter with quality presets."""

    def __init__(self):
        self.settings = get_settings()
        self._models = None  # Lazy load

    def _get_models(self) -> dict:
        """Lazy-load models (expensive operation)."""
        if self._models is None:
            self._models = create_model_dict()
        return self._models

    def convert(
        self,
        file_path: str,
        quality: str = "balanced",
        markdown_only: bool = False,
        langs: list[str] | None = None,
    ) -> dict:
        preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["balanced"])

        config = {
            "force_ocr": preset["force_ocr"],
            "use_llm": preset["use_llm"],
            "batch_multiplier": preset["batch_multiplier"],
            "languages": langs or ["English"],
        }

        converter = PdfConverter(
            artifact_dict=self._get_models(),
            config=config,
        )

        # Build document once, render to multiple formats
        document = converter.build_document(file_path)

        md_renderer = converter.resolve_dependencies(MarkdownRenderer)
        markdown = md_renderer(document).markdown

        if markdown_only:
            return {"markdown": markdown, "chunks": None}

        chunk_renderer = converter.resolve_dependencies(ChunkRenderer)
        chunks = [
            block.model_dump(mode="json", exclude={"images"})
            for block in chunk_renderer(document).blocks
        ]

        return {"markdown": markdown, "chunks": chunks}


# Singleton for Celery workers
_processor: MarkerProcessor | None = None

def get_processor() -> MarkerProcessor:
    global _processor
    if _processor is None:
        _processor = MarkerProcessor()
    return _processor
```

### Step 3: Celery Task with Time Limits

Define the async task with appropriate limits for GPU workloads:

```python
# app/tasks.py
from celery import Celery
from app.config import get_settings
from app.processor import get_processor

settings = get_settings()

celery = Celery(
    "marker",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    task_track_started=True,
    task_soft_time_limit=600,   # 10 min soft limit
    task_time_limit=900,        # 15 min hard limit
    worker_prefetch_multiplier=1,  # One task at a time for GPU
)


@celery.task(bind=True, name="convert_document")
def convert_document(
    self,
    file_path: str,
    quality: str = "balanced",
    markdown_only: bool = False,
    langs: list[str] | None = None,
) -> dict:
    processor = get_processor()
    try:
        result = processor.convert(file_path, quality, markdown_only, langs)
        return {"status": "completed", "result": result}
    except Exception as e:
        return {"status": "failed", "error": str(e)}
```

### Step 4: FastAPI Endpoints

Expose async job submission and status checking:

```python
# app/main.py
from celery.result import AsyncResult
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Marker Document Processing API")


class ConvertRequest(BaseModel):
    file_path: str
    quality: str = "balanced"
    markdown_only: bool = False
    langs: list[str] = ["English"]


@app.post("/convert")
async def submit_convert(request: ConvertRequest):
    """Submit a document for conversion."""
    if request.quality not in QUALITY_PRESETS:
        raise HTTPException(400, "Invalid quality preset")

    task = convert_document.delay(
        file_path=request.file_path,
        quality=request.quality,
        markdown_only=request.markdown_only,
        langs=request.langs,
    )
    return {"job_id": task.id, "status": "pending"}


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get the status and result of a conversion job."""
    result = AsyncResult(job_id, app=celery)

    if result.state == "PENDING":
        return {"job_id": job_id, "status": "pending"}
    elif result.state == "STARTED":
        return {"job_id": job_id, "status": "processing"}
    elif result.state == "SUCCESS":
        return {"job_id": job_id, "status": "completed", "result": result.result}
    elif result.state == "FAILURE":
        return {"job_id": job_id, "status": "failed", "error": str(result.result)}


@app.get("/health")
async def health_check():
    """Check GPU status and queue depth."""
    # Check nvidia-smi for GPU availability
    # Check Celery inspect for queue status
    ...
```

### Step 5: Docker Compose with GPU Support

Configure multi-container setup with GPU passthrough:

```yaml
# docker-compose.yml
services:
  marker-api:
    build:
      context: .
      dockerfile: Dockerfile.gpu
    ports:
      - "8001:8001"
    volumes:
      - ./data/input:/data/input
      - ./data/output:/data/output
      - marker-cache:/root/.cache/huggingface
    environment:
      - TORCH_DEVICE=cuda
      - CELERY_BROKER_URL=redis://redis:6379/0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: uvicorn app.main:app --host 0.0.0.0 --port 8001

  marker-worker:
    build:
      context: .
      dockerfile: Dockerfile.gpu
    volumes:
      - ./data/input:/data/input
      - marker-cache:/root/.cache/huggingface
    environment:
      - TORCH_DEVICE=cuda
      - CELERY_BROKER_URL=redis://redis:6379/0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: celery -A app.tasks worker --loglevel=info --concurrency=1

  redis:
    image: redis:7-alpine
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]

  flower:
    image: mher/flower:2.0
    ports:
      - "5556:5555"
    command: celery --broker=redis://redis:6379/0 flower
```

## Complete Example

Client usage from the main application:

```python
import httpx

async def process_document(file_path: str, quality: str = "balanced") -> dict:
    """Submit a document for processing and poll for result."""
    async with httpx.AsyncClient() as client:
        # Submit job
        response = await client.post(
            "http://localhost:8001/convert",
            json={"file_path": file_path, "quality": quality}
        )
        job_id = response.json()["job_id"]

        # Poll for completion
        while True:
            status = await client.get(f"http://localhost:8001/jobs/{job_id}")
            data = status.json()

            if data["status"] == "completed":
                return data["result"]
            elif data["status"] == "failed":
                raise Exception(data["error"])

            await asyncio.sleep(2)
```

## Consequences

### Benefits

- **GPU acceleration**: 10-50x faster than CPU for OCR-heavy documents
- **Quality presets**: Easy trade-off between speed and quality
- **Async processing**: Non-blocking for long documents
- **RAG-ready output**: Structured chunks with metadata
- **Monitoring**: Flower dashboard for job visibility

### Trade-offs

- **GPU requirement**: Needs NVIDIA GPU with sufficient VRAM (8GB minimum)
- **Container complexity**: Requires nvidia-container-toolkit
- **Model loading time**: First request is slow (~30s) for model warmup
- **Storage**: Models cached in Docker volume (~4GB)

### Alternatives

- **PyMuPDF + pytesseract**: Simpler but lower quality
- **AWS Textract / Google Document AI**: Managed services, pay per page
- **Unstructured.io**: Comprehensive but heavier weight

## Related Patterns

- [Centralized Environment Configuration](../stores/centralized-env-config.md) - Environment variables for service hosts

## Known Uses in Thala

- `services/marker/`: Complete GPU document processing service
- `services/services.sh`: Service orchestration with GPU detection
- `.env.example`: THALA_MARKER_HOST and GOOGLE_API_KEY configuration

## References

- [Marker PDF documentation](https://github.com/VikParuchuri/marker)
- [Celery best practices](https://docs.celeryq.dev/en/stable/userguide/tasks.html)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
