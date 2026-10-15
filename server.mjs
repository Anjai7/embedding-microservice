// embedding-service/server.mjs
// Standalone high-performance local neural embedding microservice.
// Powered by Hugging Face ONNX Transformers (@xenova/transformers).
import http from "node:http";
import { pipeline } from "@xenova/transformers";

const PORT = Number(process.env.PORT || 5005);
const MODEL_NAME = process.env.MODEL_NAME || "Xenova/all-MiniLM-L6-v2";

console.log(`[EmbeddingService] Initializing ${MODEL_NAME}...`);
const startTime = Date.now();
const embedder = await pipeline("feature-extraction", MODEL_NAME);
console.log(`[EmbeddingService] Model loaded and ready in ${Date.now() - startTime}ms!`);

const server = http.createServer(async (req, res) => {
  // Global CORS
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");

  if (req.method === "OPTIONS") {
    res.writeHead(204);
    res.end();
    return;
  }

  const url = new URL(req.url, `http://${req.headers.host || "localhost"}`);

  // GET /health
  if (req.method === "GET" && url.pathname === "/health") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(
      JSON.stringify({
        status: "ok",
        model: MODEL_NAME,
        dimensions: 384,
        uptimeSeconds: Math.round(process.uptime()),
      })
    );
    return;
  }

  // Helper to read JSON request body
  const readJsonBody = () =>
    new Promise((resolve, reject) => {
      let data = "";
      req.on("data", (chunk) => (data += chunk));
      req.on("end", () => {
        try {
          resolve(data ? JSON.parse(data) : {});
        } catch (e) {
          reject(e);
        }
      });
      req.on("error", reject);
    });

  // POST /embed (Single text)
  if (req.method === "POST" && url.pathname === "/embed") {
    const t0 = Date.now();
    try {
      const body = await readJsonBody();
      const text = typeof body.text === "string" ? body.text.trim() : "";

      if (!text) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "Missing or empty 'text' field" }));
        return;
      }

      const output = await embedder(text, { pooling: "mean", normalize: true });
      const vector = Array.from(output.data);

      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(
        JSON.stringify({
          embedding: vector,
          dimensions: vector.length,
          latencyMs: Date.now() - t0,
        })
      );
    } catch (err) {
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: err.message }));
    }
    return;
  }

  // POST /embed/batch (Multiple texts)
  if (req.method === "POST" && url.pathname === "/embed/batch") {
    const t0 = Date.now();
    try {
      const body = await readJsonBody();
      const texts = Array.isArray(body.texts) ? body.texts : [];

      if (texts.length === 0) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "Missing or empty 'texts' array" }));
        return;
      }

      const cleanTexts = texts.map((t) => (typeof t === "string" && t.trim() ? t.trim() : "empty"));
      const output = await embedder(cleanTexts, { pooling: "mean", normalize: true });

      // output.data contains all vectors flattened: [numTexts * 384]
      const dims = 384;
      const flat = Array.from(output.data);
      const embeddings = [];
      for (let i = 0; i < cleanTexts.length; i++) {
        embeddings.push(flat.slice(i * dims, (i + 1) * dims));
      }

      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(
        JSON.stringify({
          embeddings,
          count: embeddings.length,
          dimensions: dims,
          latencyMs: Date.now() - t0,
        })
      );
    } catch (err) {
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: err.message }));
    }
    return;
  }

  // 404 Not Found
  res.writeHead(404, { "Content-Type": "application/json" });
  res.end(JSON.stringify({ error: "Endpoint not found" }));
});

server.listen(PORT, () => {
  console.log(`[EmbeddingService] 🚀 Running on http://localhost:${PORT}`);
  console.log(`[EmbeddingService] Endpoints: GET /health | POST /embed | POST /embed/batch`);
});
