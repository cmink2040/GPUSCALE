# GPUSCALE

We are running and maintaining a full benchmark suite for AI-related GPU tasks. These are to showcase GPU performance similar to Blender Open Data to inform AI providers and researchers when purchasing or renting GPUs at scale, as well as maintaining a reference for future accelerators.

## Architecture

```
                        ┌──────────────┐
                        │  HuggingFace │  Public models pulled
                        │  Hub         │  directly by runner
                        └──────┬───────┘
                               │
┌──────────────┐     ┌─────────▼────┐     ┌──────────────┐     ┌──────────────┐
│  s3-attach   │     │ virt-runner  │     │    dbops     │     │ results-disp │
│              │     │              │     │              │     │              │
│ Private/gated│────▶│ Provision,   │────▶│ Submit       │────▶│ Public       │
│ models only  │     │ benchmark,   │     │ results to   │     │ leaderboard  │
│ (Wasabi S3)  │     │ teardown     │     │ Supabase     │     │ (read-only)  │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

## Components

### `s3-attach` — Private Model Pool Manager

Manages models on Wasabi S3 that **cannot be pulled directly from public sources** — Meta's original Llama weights (license-gated), custom fine-tunes, or access-restricted models.

Public models on HuggingFace (community GGUF quants, GPTQ repos, etc.) are **not** mirrored to S3. Benchmark runners pull those directly from HuggingFace Hub at runtime.

**What goes on S3 vs HuggingFace:**
| Storage | When to use | Examples |
|---------|-------------|----------|
| `s3` | Private, gated, or Meta-distributed weights | Meta Llama full weights, custom fine-tunes |
| `huggingface` | Public repos, freely downloadable | bartowski GGUF quants, community GPTQ repos, Mistral weights |

Each format in `models.toml` specifies its `storage` — `s3-attach sync` only processes formats marked `storage = "s3"`.

**Responsibilities:**
- Download gated/private models from Meta's LLaMA distribution or HuggingFace (with auth)
- Upload to a Wasabi S3 bucket in a structured layout
- Clean up local downloads after upload
- Track which models/formats are available via a manifest

**Bucket layout** (only private/gated models):
```
s3://gpuscale-models/
├── meta-llama/
│   └── Llama-3.1-8B-Instruct/
│       └── full/                # Meta-distributed FP16/BF16 weights
├── (custom models would go here)
└── manifest.json                # Index of S3-hosted models + checksums
```

**Supported formats** (expandable):
| Format | Description |
|--------|-------------|
| Full-weight | FP16/BF16 original weights |
| GGUF | llama.cpp quantized formats (Q4_K_M, Q5_K_M, Q8_0, etc.) |
| GPTQ | GPU-optimized post-training quantization |

**Supported models** are defined in `s3-attach/models.toml`. Each format specifies `storage = "s3"` or `storage = "huggingface"` to control where it lives.

### `virt-runner` — Benchmark Orchestrator

The core benchmarking component. Handles the full lifecycle: provision infrastructure, run standardized benchmarks inside a container, collect results, tear down.

**Deployment approach:** SSH + provider CLIs for orchestration, Docker only for the benchmark execution environment.

Cloud GPU providers (Vast.ai, RunPod) give you SSH access to instances with NVIDIA drivers, CUDA, and Docker pre-installed. There's no reason to build custom deployment infrastructure — use the provider tools (`vastai`, RunPod API) for instance lifecycle and SSH for everything else. The Docker container's role is benchmark isolation, not deployment.

```
virt-runner (local machine)
  │
  ├─ vastai create instance / runpod create pod   # Provision via provider CLI/API
  ├─ SSH into instance
  ├─ docker pull gpuscale-bench                   # Pull benchmark container ON the instance
  ├─ docker run ... benchmark                     # Run benchmark inside container
  ├─ scp results back                             # Collect results
  └─ vastai destroy instance / runpod stop pod    # Teardown via provider CLI/API
```

**Targets:**
- **Cloud:** Vast.ai, RunPod — provision via provider API/CLI, SSH in, run containerized benchmark, teardown
- **Local:** Run benchmark container directly on host GPU(s)

**Workflow (cloud):**
```
1. Select GPU type + provider from job config
2. Provision instance via provider CLI/API (vastai / RunPod)
3. SSH into the instance
4. Pull benchmark container on the instance
5. Container pulls model from Wasabi S3
6. Run benchmark workload (standardized prompt set)
7. Collect metrics, scp results back to host
8. Tear down instance via provider CLI/API
```

**Workflow (local):**
```
1. Detect local GPU(s) via nvidia-smi / rocm-smi
2. Collect host environment metadata (OS, kernel, driver version)
3. Run benchmark container locally (docker run --gpus all)
4. Container pulls model from Wasabi S3 (or use local cache)
5. Run benchmark workload
6. Collect metrics
7. Return results with host metadata attached
```

**OS and environment handling:**

Cloud instances are always Linux — the container normalizes the software environment, so results from Vast.ai and RunPod are directly comparable.

Local benchmarking is **not** OS-agnostic. The host OS, kernel version, and driver version can all affect GPU performance. Local results record:
- OS and distribution (e.g. `Ubuntu 24.04`, `Windows 11 + WSL2`, `Arch Linux`)
- Kernel version (e.g. `6.8.0-45-generic`)
- NVIDIA driver version on host (e.g. `550.54.14`)
- Docker runtime (e.g. `nvidia-container-toolkit 1.16.1`)

This metadata is stored alongside the benchmark results so that local submissions from different OSes are distinguishable and filterable, not falsely treated as equivalent.

**Benchmark container:**

The benchmark runs inside a standardized Docker container. The container pins:
- Inference engines: llama.cpp, vLLM (more can be added)
- CUDA runtime version
- Metric collection tooling (nvidia-smi polling, engine-native stats)
- The benchmark harness itself

The container ensures the software stack is identical across runs — what varies is the hardware and (for local runs) the host OS environment.

**Inference engines:**
| Engine | Use case |
|--------|----------|
| llama.cpp | CPU/GPU inference, GGUF models, single-GPU consumer hardware |
| vLLM | GPU inference, full-weight/GPTQ models, multi-GPU, production-style serving |

Each benchmark result is tagged with the engine used, enabling direct comparisons across engines on the same hardware.

**Metrics collected:**

| Metric | Source | Description |
|--------|--------|-------------|
| Tokens/sec (generation) | Engine stats | Sustained token generation throughput |
| Time-to-first-token (TTFT) | Engine stats | Latency from prompt submission to first generated token |
| Prompt eval rate | Engine stats | Tokens/sec during prompt processing (prefill) |
| Peak VRAM usage | nvidia-smi | High-water mark GPU memory during inference |
| GPU TDP / Power draw | nvidia-smi | Power consumption during the benchmark |
| GPU utilization % | nvidia-smi | Average and peak GPU core utilization |
| GPU temperature | nvidia-smi | Temperature during benchmark |
| Total wall time | Harness | End-to-end time for the full benchmark run |

**Benchmark workload:**

The default workload is a standardized set of prompts with fixed generation parameters (temperature, max tokens, etc.) to ensure comparability. This is configurable — the workload definition and parameters are stored as metadata alongside the results, so custom runs remain interpretable.

Default workload spec (example):
```json
{
  "workload_version": "1.0",
  "prompts": [
    {"role": "user", "content": "...standardized prompt 1..."},
    {"role": "user", "content": "...standardized prompt 2..."}
  ],
  "generation_params": {
    "max_tokens": 512,
    "temperature": 0.0,
    "top_p": 1.0
  },
  "iterations": 5,
  "warmup_iterations": 1
}
```

### `dbops` — Database Operations CLI

A CLI tool with write access to the Supabase database. This is the only path for submitting benchmark results — it validates, formats, and inserts results.

**Responsibilities:**
- Submit benchmark results (from virt-runner output)
- Validate result schema before insertion
- Tag results with metadata (hardware, engine, model, workload version, timestamp, provider)
- Admin operations (manage models list, flag/remove bad results)

The database itself (Supabase/Postgres) is publicly readable but write-restricted. `dbops` authenticates with a service role key.

**Core tables (draft schema):**

```
benchmark_results
├── id (uuid)
├── created_at (timestamp)
├── gpu_name (text)              # e.g. "NVIDIA RTX 4090"
├── gpu_vram_gb (numeric)        # e.g. 24
├── gpu_count (int)              # e.g. 1
├── provider (text)              # "local", "vast.ai", "runpod"
├── engine (text)                # "llama.cpp", "vllm"
├── model_name (text)            # e.g. "meta-llama/Llama-3.1-8B-Instruct"
├── quantization (text)          # "Q4_K_M", "GPTQ-4bit", "FP16"
├── workload_version (text)      # e.g. "1.0"
├── workload_config (jsonb)      # Full workload spec for reproducibility
├── tokens_per_sec (numeric)
├── time_to_first_token_ms (numeric)
├── prompt_eval_tokens_per_sec (numeric)
├── peak_vram_mb (numeric)
├── avg_power_draw_w (numeric)
├── peak_power_draw_w (numeric)
├── avg_gpu_util_pct (numeric)
├── avg_gpu_temp_c (numeric)
├── total_wall_time_s (numeric)
├── engine_version (text)        # e.g. "llama.cpp b4567"
├── host_os (text)               # e.g. "Ubuntu 24.04", "Windows 11 + WSL2" (local only)
├── host_kernel (text)           # e.g. "6.8.0-45-generic" (local only)
├── host_driver_version (text)   # e.g. "NVIDIA 550.54.14" (host-level)
├── container_image (text)       # Docker image tag used
├── container_driver_version (text) # Driver version inside container
└── raw_output (jsonb)           # Full engine output for debugging
```

### `results-disp` — Public Leaderboard

A Next.js web app that reads from the public Supabase database and displays benchmark results.

**Initial scope (kept simple — refine later):**
- Filterable table: GPU, model, engine, quantization, provider
- Sort by any metric column
- Basic comparison view (e.g. GPU A vs GPU B on the same model)

The data pipeline is the hard part — visualization can be iterated on once there's real data flowing.

## Tech Stack

| Component | Stack |
|-----------|-------|
| s3-attach | Python, uv, boto3 (S3-compatible) |
| virt-runner | Python, uv, Docker, provider APIs |
| dbops | Python, uv, supabase-py |
| results-disp | Next.js, TypeScript, Supabase JS client |
| Database | Supabase (Postgres) |
| Model storage | Wasabi S3 |
| Containers | Docker |

## Environment Variables

See `.env.example` for required keys:

```
VAST_API_KEY=         # Vast.ai API key
RUNPOD_API_KEY=       # RunPod API key
WASABI_ACCESS_KEY=    # Wasabi S3 access key
WASABI_SECRET_KEY=    # Wasabi S3 secret key
WASABI_BUCKET=        # S3 bucket name
SUPABASE_URL=         # Supabase project URL
SUPABASE_PUBLISHABLE_KEY= # Supabase publishable key (read-only, for results-disp)
SUPABASE_SECRET_KEY=      # Supabase secret key (write access, for dbops)
DATABASE_URL=             # Supabase Postgres connection string (for dbops/Alembic)
```

## Open Design Questions

- **Container strategy:** Single fat container with all engines, or per-engine images? Fat container is simpler to deploy (one pull, one image to manage) but larger. Per-engine images are leaner but add orchestration complexity.
- **Result verification:** How to prevent fabricated submissions? Options: signed container attestation, require raw nvidia-smi logs, hash-based verification of workload output.
- **Multi-GPU benchmarking:** vLLM supports tensor parallelism — should multi-GPU configs be a first-class benchmark target?

## License

MIT
