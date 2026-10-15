# Standalone Neural Embedding Microservice

High-performance, standalone, 100% self-hosted neural text embedding microservice powered by Hugging Face ONNX Transformers (`@xenova/transformers`).

## Model
* **Model:** `Xenova/all-MiniLM-L6-v2` (6-layer BERT Transformer with 22.7M parameters)
* **Vector Dimensions:** 384
* **Latency:** ~6ms - 15ms per embedding
* **Cost:** $0.00 / 100% Free / Zero API keys / Zero rate limits

## Endpoints

### 1. `GET /health`
Returns service status and active model.
```json
{
  "status": "ok",
  "model": "Xenova/all-MiniLM-L6-v2",
  "dimensions": 384,
  "uptimeSeconds": 120
}
```

### 2. `POST /embed`
Generate embedding for a single text string.
```json
// Request
{
  "text": "Outdoorsy hiking and mountain climbing"
}

// Response
{
  "embedding": [-0.021, 0.045, ...],
  "dimensions": 384,
  "latencyMs": 8
}
```

### 3. `POST /embed/batch`
Generate embeddings for a batch of texts.
```json
// Request
{
  "texts": ["Coffee lover", "Tech founder", "Artist"]
}

// Response
{
  "embeddings": [[...], [...], [...]],
  "count": 3,
  "dimensions": 384,
  "latencyMs": 15
}
```

## Running Locally
```bash
npm install
npm start
```
Default port: `5005` (configurable via `PORT` environment variable).
