/**
 * Pluggable embedding engine.
 *
 * Default is a fully-free, offline, deterministic local embedder (n-gram hashing
 * trick + L2-normalized dense vector). It needs no network, no API key, and no
 * native modules, so it works anywhere Node runs.
 *
 * For materially better semantic similarity you can opt into a real embedding
 * model without changing the API surface:
 *   - Ollama (local + free):      EMBED_BACKEND=ollama   (defaults to http://localhost:11434, model nomic-embed-text)
 *   - OpenAI-compatible endpoint: EMBED_BACKEND=openai   (needs OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_EMBED_MODEL)
 */

export interface Embedder {
  readonly dim: number;
  embed(text: string): Promise<number[]>;
  label: string;
}

/* ------------------------------------------------------------------ */
/* Local hashing embedder (offline, deterministic, dependency-free)    */
/* ------------------------------------------------------------------ */

const DIM = 384;

function tokenize(text: string): string[] {
  return (text.toLowerCase().match(/[a-z0-9_]+/g) ?? []).filter(Boolean);
}

function ngrams(tokens: string[], n: number): string[] {
  const out: string[] = [];
  for (let i = 0; i + n <= tokens.length; i++) out.push(tokens.slice(i, i + n).join('_'));
  return out;
}

/** Stable 32-bit FNV-1a hash -> bucket index + signed weight. */
function hashToBucket(str: string): { idx: number; sign: number } {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  h >>>= 0;
  return { idx: h % DIM, sign: h & 1 ? 1 : -1 };
}

class LocalEmbedder implements Embedder {
  readonly dim = DIM;
  readonly label = 'local-hashing-v1';

  async embed(text: string): Promise<number[]> {
    const tokens = tokenize(text);
    // Use a plain number array directly — avoids Float64Array → Array.from() copy.
    const vec: number[] = new Array<number>(DIM).fill(0);
    // character trigrams capture formatting quirks (param shapes, filenames)
    const raw = text.toLowerCase().replace(/\s/g, '');
    for (let i = 0; i + 3 <= raw.length; i++) {
      const b = hashToBucket(raw.slice(i, i + 3));
      vec[b.idx] += b.sign;
    }
    for (const tok of tokens) {
      const b = hashToBucket(tok);
      vec[b.idx] += 2 * b.sign;
    }
    for (const gram of ngrams(tokens, 1)) {
      const b = hashToBucket(`g1:${gram}`);
      vec[b.idx] += b.sign;
    }
    for (const gram of ngrams(tokens, 2)) {
      const b = hashToBucket(`g2:${gram}`);
      vec[b.idx] += 2 * b.sign;
    }
    return normalize(vec);
  }
}

/* ------------------------------------------------------------------ */
/* Ollama (local, free)                                                */
/* ------------------------------------------------------------------ */

class OllamaEmbedder implements Embedder {
  readonly dim = 0;
  readonly label = 'ollama/nomic-embed-text';
  private base: string;
  private model: string;

  constructor(base?: string, model?: string) {
    this.base = (base ?? process.env.OLLAMA_BASE_URL ?? 'http://localhost:11434').replace(/\/$/, '');
    this.model = model ?? process.env.OLLAMA_EMBED_MODEL ?? 'nomic-embed-text';
  }

  async embed(text: string): Promise<number[]> {
    const res = await fetch(`${this.base}/api/embeddings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: this.model, prompt: text }),
    });
    if (!res.ok) throw new Error(`Ollama embeddings failed: ${res.status} ${await res.text()}`);
    const data = (await res.json()) as { embedding: number[] };
    return normalize(data.embedding);
  }
}

/* ------------------------------------------------------------------ */
/* OpenAI-compatible                                                   */
/* ------------------------------------------------------------------ */

class OpenAIEmbedder implements Embedder {
  readonly dim = 0;
  readonly label = 'openai/text-embedding-3-small';
  private base: string;
  private model: string;
  private key: string;

  constructor() {
    this.key = process.env.OPENAI_API_KEY ?? '';
    this.base = (process.env.OPENAI_BASE_URL ?? 'https://api.openai.com/v1').replace(/\/$/, '');
    this.model = process.env.OPENAI_EMBED_MODEL ?? 'text-embedding-3-small';
  }

  async embed(text: string): Promise<number[]> {
    if (!this.key) throw new Error('OPENAI_API_KEY is not set');
    const res = await fetch(`${this.base}/embeddings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${this.key}` },
      body: JSON.stringify({ model: this.model, input: text }),
    });
    if (!res.ok) throw new Error(`OpenAI embeddings failed: ${res.status} ${await res.text()}`);
    const data = (await res.json()) as { data: { embedding: number[] }[] };
    return normalize(data.data[0].embedding);
  }
}

/* ------------------------------------------------------------------ */

export function normalize(vec: number[]): number[] {
  let sum = 0;
  for (const v of vec) sum += v * v;
  const norm = Math.sqrt(sum) || 1;
  return vec.map((v) => v / norm);
}

/** Euclidean distance between two equal-length vectors. */
export function distance(a: number[], b: number[]): number {
  let s = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return Math.sqrt(s);
}

/** Cosine similarity between two normalized vectors in [-1, 1]. */
export function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) dot += a[i] * b[i];
  return dot;
}

export function buildEmbedder(): Embedder {
  const backend = (process.env.EMBED_BACKEND ?? 'local').toLowerCase();
  switch (backend) {
    case 'ollama':
      return new OllamaEmbedder();
    case 'openai':
      return new OpenAIEmbedder();
    case 'local':
      return new LocalEmbedder();
    default:
      throw new Error(`Unknown EMBED_BACKEND "${backend}". Use "local", "ollama" or "openai".`);
  }
}