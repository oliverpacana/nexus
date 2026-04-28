```
  ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗
  ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝
  ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗
  ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║
  ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║
  ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝
```

# Nexus — AI Agent Operating System

**The runtime layer that AI agents have always needed. Built in Rust.**

[![CI](https://github.com/nexus-runtime/nexus/actions/workflows/ci.yml/badge.svg)](https://github.com/nexus-runtime/nexus/actions)
[![Crates.io](https://img.shields.io/crates/v/nexus-cli.svg)](https://crates.io/crates/nexus-cli)
[![Docs.rs](https://docs.rs/nexus-proto/badge.svg)](https://docs.rs/nexus-proto)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust: 1.79+](https://img.shields.io/badge/rust-1.79%2B-orange.svg)](https://www.rust-lang.org)
[![Discord](https://img.shields.io/discord/000000000?label=discord&logo=discord)](https://discord.gg/nexus-runtime)

---

## What Is Nexus?

Every AI agent framework today is a **library**. They give you helper functions to chain prompts, call tools, and maybe run two agents at once. But when your agent crashes mid-task, it takes everything down with it. When two agents need to share a discovery, you pass a dict. When a runaway agent spends $200 in tokens overnight, there is no kill switch.

**Nexus is a runtime, not a library.** The distinction is the same one that separates a kernel from an application. When you spawn an agent in Nexus, it gets a supervised process boundary, an isolated memory scope, a declared capability set, a scheduler slot, and a full distributed trace — automatically. You write the agent logic. Nexus runs it safely, observably, and at scale.

The thesis: AI agents are being built like 1970s programs — single-threaded, stateless, fragile. Nexus treats agents like first-class operating system processes. It is to AI agents what `tokio` is to async Rust: the runtime that makes everything else possible.

---

## Why Rust?

- **Zero-cost async** — tokens stream through the pipeline with no GC pauses, no stop-the-world events, no latency spikes mid-response
- **Memory safety without a runtime** — agents manage sensitive context, tool outputs, and API keys; Nexus makes undefined behavior structurally impossible
- **WebAssembly-first tooling** — every tool plugin compiles to `.wasm` via the same `cargo` workflow; isolation is free
- **True concurrency** — `tokio` gives us M:N threading that can run 50 concurrent agents on a laptop without burning through file descriptors
- **`crates.io` ecosystem** — `wasmtime`, `libp2p`, `usearch`, `ratatui`, `tiktoken-rs` — every critical dependency already exists, is production-grade, and is maintained

---

## Architecture

Nexus is composed of seven cooperating subsystems. Each is an independent crate with a clean public API. They share only the types defined in `nexus-proto`.

```
┌─────────────────────────────────────────────────────────────────────┐
│                           nexus-cli                                  │
│              up / run / status / tui / mem / tool                   │
└─────────────────────────────────┬───────────────────────────────────┘
                                   │  bootstraps and owns
┌──────────────────────────────────▼───────────────────────────────────┐
│                          nexus-kernel                                 │
│                                                                       │
│   AgentProcess ──── SupervisorTree ──── Scheduler ──── CapGuard      │
│                                                                       │
│   Spawn / Kill / Suspend / Resume │ Capability enforcement           │
│   OneForOne / OneForAll / RestForOne restart strategies              │
│   Token-bucket rate limiting per agent                               │
└───────┬──────────────────┬────────────────────┬──────────────────────┘
        │                  │                    │
┌───────▼──────┐  ┌────────▼────────┐  ┌────────▼─────────────┐
│  nexus-mem   │  │  nexus-tools    │  │    nexus-router       │
│              │  │                 │  │                       │
│  L1 Working  │  │  wasmtime       │  │  OpenAI   Anthropic   │
│  L2 Episodic │  │  WASM sandbox   │  │  Groq     Mistral     │
│  L3 Semantic │  │  Per-call iso.  │  │  Local    Custom      │
│  L4 Procedur │  │  Hot reload     │  │                       │
└──────┬───────┘  └────────┬────────┘  └────────┬─────────────┘
       │                   │                    │
┌──────▼───────────────────▼────────────────────▼──────────────────────┐
│                           nexus-flow                                  │
│                                                                       │
│   WorkflowDag ──── StepExecutor ──── Checkpointing ──── DSL          │
│   Parallel branches │ Conditional routing │ Resume from crash        │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────────────┐
│                           nexus-mesh                                  │
│                                                                       │
│   libp2p transport ──── CRDT Blackboard ──── Capability Discovery    │
│   mDNS LAN discovery │ P2P WAN mesh │ Partition-tolerant sync        │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────────────┐
│                           nexus-obs                                   │
│                                                                       │
│   tokio-tracing ──── OpenTelemetry ──── CostLedger ──── Replay       │
│   Per-token tracing │ OTLP export │ Per-agent cost │ Session replay  │
└─────────────────────────────────────────────────────────────────────┘

                    Shared foundation: nexus-proto
            (AgentId, ModelRequest, ToolCall, MemoryEntry, …)
```

---

## Quick Start

### Install

```bash
# From crates.io
cargo install nexus-cli

# Or build from source
git clone https://github.com/nexus-runtime/nexus
cd nexus
cargo build --release
cp target/release/nexus ~/.local/bin/
```

### Configure a Provider

Nexus needs at least one LLM provider. Set the key for whichever you have:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENAI_API_KEY=sk-...
# or
export GROQ_API_KEY=gsk_...
# or start Ollama locally — nexus auto-detects it at http://localhost:11434
```

### Start the Runtime

```bash
nexus up
```

```
  ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗
  ...

  ⚡ Nexus v0.1.0 — AI Agent Runtime
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Starting subsystems...
  ✓ Memory        (L1 working / L2 episodic / L3 semantic / L4 procedural)
  ✓ Tools         (3 built-in tools loaded: web-search, http-fetch, code-exec)
  ✓ Router        (provider: Anthropic claude-3-5-sonnet-20241022)
  ✓ Kernel        (max agents: 256, scheduler: priority + token-bucket)
  ✓ Observability (cost ledger: ~/.local/share/nexus/costs.db)

  Runtime ready. Press Ctrl+C to shutdown.
```

### Run Your First Workflow

```bash
nexus run examples/workflows/research-and-write.toml \
  --var topic="the future of Rust in systems programming" \
  --var word_count=2000
```

### Open the Live Dashboard

```bash
nexus tui
```

```
⚡ NEXUS  ── Tuesday 2026-04-28 14:32:01 ──────────────────────── v0.1.0
┌ Agents ──────────────────────────────────────────────────────────────┐
│ [Agents]  [Cost]  [Logs]                                             │
├──────────────────────────────────────────────────────────────────────┤
│ ● researcher-7f3a    Research    High      running    0:00:34        │
│ ◌ writer-2b19        Writing     Normal    waiting    —              │
│ ◌ editor-9c44        Analysis    Normal    waiting    —              │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│ Peers: 0   Active: 1/256   Cost today: $0.0023   Uptime: 0:01:12    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Configuration

Nexus is configured through `nexus.toml`. It looks in these locations in order:

1. `--config-file <PATH>` flag
2. `./nexus.toml` (current directory)
3. `~/.config/nexus/nexus.toml`
4. `/etc/nexus/nexus.toml`

Copy the annotated example to get started:

```bash
cp config/nexus.example.toml ./nexus.toml
```

The most important sections:

```toml
[providers.anthropic]
api_key_env = "ANTHROPIC_API_KEY"
default_model = "claude-3-5-sonnet-20241022"

[providers.openai]
api_key_env = "OPENAI_API_KEY"
default_model = "gpt-4o"

[providers.local]
enabled = true
base_url = "http://localhost:11434/v1"
default_model = "llama3.2"

[router]
# Route each request to the cheapest provider that meets the latency SLA
default_policy = "cost_optimized"
default_max_latency_ms = 5000

[memory]
episodic_db_path = "~/.local/share/nexus/episodic.db"
semantic_index_path = "~/.local/share/nexus/semantic.idx"
embedding_provider = "local"   # no API key needed for embeddings

[mesh]
enabled = false   # set true to join a distributed agent mesh

[observability]
log_level = "info"
log_format = "pretty"
# otlp_endpoint = "http://localhost:4318"  # uncomment for Jaeger/Grafana
```

---

## The Seven Subsystems

### 1. `nexus-kernel` — Agent Process Manager

The kernel manages the complete lifecycle of every agent: spawn, suspend, resume, kill. Agents are Tokio tasks wrapped in a supervision tree. When an agent fails, the supervisor applies a restart strategy — `OneForOne` (restart only the failed agent), `OneForAll` (restart all siblings), or `RestForOne` (restart the failed agent and everyone started after it). This is borrowed directly from Erlang/OTP and makes fault isolation a first-class primitive.

Every agent declares a capability set at spawn time: which tools it can call, which memory tiers it can write to, which model providers it can access, whether it can spawn children. The kernel enforces these at every boundary. An agent that attempts to write to semantic memory without the capability gets `CapabilityDenied` — not a silent failure, not a runtime panic.

The scheduler assigns each agent a priority (`Critical` through `Background`) and enforces per-agent token-bucket rate limiting. A runaway agent cannot saturate your LLM API or starve other agents of scheduler time.

### 2. `nexus-mem` — Four-Tier Memory Hierarchy

Memory is the hardest unsolved problem in AI agents. Nexus models it as a four-tier cache hierarchy analogous to CPU caches:

| Tier | Analogy | Backend | Scope | Speed |
|---|---|---|---|---|
| **L1 Working** | CPU register | `Arc<RwLock<HashMap>>` | Per-agent, current task | Nanoseconds |
| **L2 Episodic** | RAM | Embedded SQLite (`libsql`) | Per-agent lifetime, event log | Milliseconds |
| **L3 Semantic** | SSD | Vector index (`usearch` HNSW) | Cross-agent, persistent | Milliseconds |
| **L4 Procedural** | Cold storage | Knowledge graph (`sled`) | Global, versioned, structured | Milliseconds |

Memory is permissioned by scope. `Private` entries are only accessible to the owning agent. `Group` entries are shared within a supervisor group. `Global` entries are accessible to any agent in the mesh. Writes to non-owned scopes require explicit grants from the kernel. L2 episodic memory is append-only and powers the replay engine — every agent run can be re-executed with bit-for-bit identical inputs for debugging.

### 3. `nexus-tools` — WASM Plugin Sandbox

Every tool in Nexus is a WebAssembly module loaded into a `wasmtime` sandbox. The sandbox enforces hard limits: a declared list of allowed hostnames for network access, a maximum memory allocation, a CPU fuel budget that kills infinite loops, and a per-call timeout. A tool cannot escape its declared capabilities regardless of what code it runs.

Tools ship with a `manifest.toml` declaring their name, version, resource limits, and JSON Schema definitions for their input and output. Every tool call is validated against these schemas before execution and after. Tool plugins support hot reloading — `nexus tool reload web-search` swaps in a new version without restarting agents or the runtime.

Nexus ships four built-in tools: `web-search` (DuckDuckGo), `http-fetch` (URL content extraction with HTML stripping), `code-exec` (in-process Python/JS/Shell/Lua interpreter), and `file-read` (sandboxed file access with CSV, JSON, and TOML parsing).

### 4. `nexus-router` — Universal Model Gateway

The router abstracts every LLM provider behind a single streaming interface. Every provider — OpenAI, Anthropic, Groq, Mistral, and local Ollama/llama.cpp — implements the `ModelProvider` trait. The trait surface is minimal: `stream_completion(request) -> impl Stream<Item = Token>`. Everything else — API key management, retry logic, error normalization, cost estimation — is handled per-provider.

Routing policies let you express intent rather than provider preference. `CostOptimized` selects the cheapest provider that can meet a latency SLA. `LocalFirst` tries your local model and falls back to cloud if unavailable or context window exceeded. `Pinned` routes to a specific model when you need determinism. Every token that flows through the router is metered and reported to the cost ledger.

### 5. `nexus-mesh` — Distributed Agent Fabric

When a single machine isn't enough, Nexus forms a peer-to-peer mesh using `libp2p`. Nodes advertise their capabilities and agents discover remote workers by capability rather than address. An agent that needs code execution doesn't ask a specific server — it asks the mesh for "any node with `tool:code-exec` and `model:gpt-4o`" and the mesh routes accordingly.

Shared state synchronizes via a CRDT-based distributed blackboard. Because CRDTs are eventually consistent by design, there's no distributed lock, no consensus round-trip, and no single point of failure. If the network partitions, each side keeps working. When it heals, the blackboards merge automatically. LAN discovery is zero-configuration via mDNS. WAN mesh requires a configurable rendezvous address.

### 6. `nexus-flow` — Workflow DAG Engine

Most real AI tasks are pipelines, not single agents. `nexus-flow` is a typed DAG executor for multi-step workflows. Each node in the DAG is a `Step` — either an agent invocation, a direct tool call, a conditional branch (driven by structured LLM output), or a set of parallel steps joined by `tokio::join_all`. Workflows are defined in a Rust builder DSL or loaded from TOML/YAML at runtime.

Before each step executes, the workflow state is checkpointed to SQLite. If the process crashes or the workflow is interrupted, `nexus run --resume <run-id>` picks up from the last completed step. Conditional routing uses structured model output — the workflow asks the model for a JSON decision, validates it against a schema, and routes to the matching branch without fragile string parsing.

### 7. `nexus-obs` — Observability Layer

Every token, every tool call, every state transition, every cost cent is observable. `tokio-tracing` instruments every subsystem with structured spans. Traces export to any OpenTelemetry collector — Jaeger, Zipkin, Grafana Tempo — via OTLP. The cost ledger records per-agent, per-model, per-session token usage and estimated USD cost, queryable by time range and exportable to CSV.

The replay engine re-executes any past agent session from its L2 episodic event log with the exact inputs it originally saw. This is the debugging superpower that no other agent framework has: "why did this agent make the wrong decision on Tuesday?" — replay it, step through it, change one variable, compare the two runs with `ReplayDiff`. The live TUI dashboard (`nexus tui`) shows the running mesh, per-agent status, cost table, and scrollable log stream, refreshing at 250ms.

---

## Building from Source

**Requirements:**
- Rust 1.79 or later (`rustup update stable`)
- For WASM tool compilation: `rustup target add wasm32-unknown-unknown`
- For mesh features on Linux: `libssl-dev`, `pkg-config`

```bash
git clone https://github.com/nexus-runtime/nexus
cd nexus

# Build everything
cargo build --release

# Run tests
cargo test --workspace

# Build WASM tools
cd tools/web-search && cargo build --target wasm32-unknown-unknown --release
cd ../http-fetch && cargo build --target wasm32-unknown-unknown --release
cd ../code-exec && cargo build --target wasm32-unknown-unknown --release
cd ../file-read && cargo build --target wasm32-unknown-unknown --release

# Copy compiled tools to registry
mkdir -p ~/.local/share/nexus/tools
cp tools/*/target/wasm32-unknown-unknown/release/*.wasm ~/.local/share/nexus/tools/
cp tools/*/manifest.toml ~/.local/share/nexus/tools/   # copy alongside each wasm

# Install the CLI
cargo install --path crates/nexus-cli
```

**Verify the build:**

```bash
nexus --version
# nexus 0.1.0

cargo clippy --workspace -- -D warnings
# should produce no errors

cargo test --workspace
# test result: ok. N passed; 0 failed
```

---

## Writing a Custom Agent

Implement the `AgentTask` trait from `nexus-kernel`. Your agent receives an `AgentContext` which is the syscall interface to everything: the router, memory, tools, the message inbox, and the shutdown signal.

```rust
use nexus_kernel::{AgentTask, AgentContext};
use nexus_proto::{AgentKind, AgentCapabilities, MemoryAccess};
use async_trait::async_trait;

pub struct SummarizerAgent {
    pub text: String,
    pub max_words: usize,
}

#[async_trait]
impl AgentTask for SummarizerAgent {
    fn name(&self) -> &str { "summarizer" }

    fn kind(&self) -> AgentKind { AgentKind::Analysis }

    fn capabilities(&self) -> AgentCapabilities {
        AgentCapabilities::new()
            .with_model("anthropic/*")
            .with_model("openai/*")
            .with_memory(MemoryAccess::Write)
    }

    async fn run(
        &mut self,
        ctx: AgentContext,
    ) -> anyhow::Result<serde_json::Value> {
        // Check shutdown signal before expensive work
        if ctx.is_shutting_down() {
            return Ok(serde_json::json!({ "cancelled": true }));
        }

        // Build a model request
        let request = ModelRequest::builder()
            .system("You are a concise summarizer.")
            .user(format!(
                "Summarize the following in under {} words:\n\n{}",
                self.max_words, self.text
            ))
            .max_tokens(self.max_words as u32 * 2)
            .build()?;

        // Route to the best available model
        let response = ctx.router().complete(request, ctx.agent_id()).await?;
        let summary = response.message.text_content();

        // Store in L1 working memory for other agents to read
        ctx.memory()
            .get_working(ctx.agent_id())
            .set("summary", serde_json::json!(summary))
            .await?;

        Ok(serde_json::json!({
            "summary": summary,
            "original_words": self.text.split_whitespace().count(),
            "summary_words": summary.split_whitespace().count(),
        }))
    }
}
```

Spawn it from the kernel:

```rust
let agent_id = kernel.spawn(
    SummarizerAgent {
        text: long_article.clone(),
        max_words: 150,
    },
    SpawnOptions {
        name: Some("summarizer-main".to_string()),
        priority: AgentPriority::Normal,
        capabilities: SummarizerAgent::capabilities_static(),
        supervisor_id: Some("my-supervisor".to_string()),
        ..Default::default()
    },
).await?;
```

---

## Writing a Custom Tool

Tools are WebAssembly modules. Write your tool in any language that compiles to WASM. The Nexus tool ABI uses static input/output buffers and a small set of host functions.

**Rust example:**

```rust
// In tools/my-tool/src/lib.rs
// Compile with: cargo build --target wasm32-unknown-unknown --release

static mut INPUT_BUFFER: [u8; 65536] = [0u8; 65536];
static mut OUTPUT_BUFFER: [u8; 262144] = [0u8; 262144];
static mut OUTPUT_LEN: u32 = 0;
static mut INPUT_LEN: u32 = 0;

extern "C" {
    fn nexus_log(level: i32, msg_ptr: i32, msg_len: i32);
}

#[no_mangle]
pub extern "C" fn nexus_get_input_ptr() -> *mut u8 {
    unsafe { INPUT_BUFFER.as_mut_ptr() }
}

#[no_mangle]
pub extern "C" fn nexus_get_input_len() -> u32 {
    unsafe { INPUT_LEN }
}

#[no_mangle]
pub extern "C" fn nexus_set_input_len(len: u32) {
    unsafe { INPUT_LEN = len; }
}

#[no_mangle]
pub extern "C" fn nexus_get_output_ptr() -> *mut u8 {
    unsafe { OUTPUT_BUFFER.as_mut_ptr() }
}

#[no_mangle]
pub extern "C" fn nexus_get_output_len() -> u32 {
    unsafe { OUTPUT_LEN }
}

#[no_mangle]
pub extern "C" fn nexus_set_output_len(len: u32) {
    unsafe { OUTPUT_LEN = len; }
}

#[no_mangle]
pub extern "C" fn _nexus_run() -> i32 {
    // Read input JSON from INPUT_BUFFER
    let input_bytes = unsafe { &INPUT_BUFFER[..INPUT_LEN as usize] };
    // ... parse, process, write output JSON to OUTPUT_BUFFER ...
    // return 0 on success, 1 on error
    0
}
```

Pair it with a `manifest.toml` and install it:

```bash
nexus tool install ./my-tool.wasm
nexus tool list
# my-tool  v1.0.0  calls: 0  errors: 0
```

See [`docs/tool-authoring.md`](docs/tool-authoring.md) for the complete ABI reference, host function signatures, and a step-by-step tutorial.

---

## CLI Reference

```
nexus <command> [options]

Commands:
  up                        Start the Nexus runtime daemon
  run <workflow.toml>       Execute a workflow file
    --var key=value         Override a workflow variable (repeatable)
    --resume <run-id>       Resume a previously interrupted run
    --show-cost             Print cost breakdown after completion
  status                    Show all running agents and system stats
  tui                       Launch the live terminal dashboard
  mem <subcommand>          Memory management
    inspect                 Show memory contents
      --agent <id>          Filter by agent
      --tier l1|l2|l3|l4   Filter by memory tier
    search <query>          Semantic search across L3 memory
    clear                   Clear memory entries
      --agent <id>          For a specific agent
      --tier <tier>         For a specific tier
  tool <subcommand>         Tool plugin management
    list                    Show installed tools with usage stats
    install <path.wasm>     Install a WASM tool
    remove <name>           Uninstall a tool
    reload <name>           Hot-reload a tool (zero downtime)

Global flags:
  --config-file <path>      Path to nexus.toml config file
  --log-level <level>       Override log level (trace/debug/info/warn/error)
  --no-color                Disable colored output
  --version                 Print version
  --help                    Print help
```

---

## Examples

The `examples/` directory contains two runnable examples and two workflow definitions:

**`examples/research-agent`** — A single specialized research agent that searches the web, fetches pages, scores relevance, stores findings in semantic memory, and synthesizes a full research report. Demonstrates tool calls, L3 memory writes, multi-step LLM reasoning, and cost reporting.

```bash
cargo run -p nexus-example-research-agent -- \
  --topic "Rust async runtime internals" \
  --depth deep \
  --show-cost \
  --output report.md
```

**`examples/multi-agent-pipeline`** — Five specialized agents working in a supervised pipeline: ResearchAgent → OutlineAgent → WriterAgent → EditorAgent → SEOAgent. Agents communicate via shared L3 semantic memory. Demonstrates supervision trees, parallel execution, live multi-progress bars, and per-agent cost accounting.

```bash
cargo run -p nexus-example-multi-agent-pipeline -- \
  --topic "WebAssembly in production systems" \
  --audience technical \
  --content-type article \
  --word-count 2500 \
  --parallel \
  --save-dir ./output
```

**`examples/workflows/research-and-write.toml`** — A TOML workflow demonstrating all step types: agent, tool, conditional branching, parallel execution, and transform steps. Nine steps from raw search to polished article.

**`examples/workflows/code-review.toml`** — A parallel code review workflow: four specialized review agents (correctness, security, performance, style) run concurrently, then a synthesis agent produces a prioritized report with fix suggestions.

```bash
nexus run examples/workflows/code-review.toml \
  --var file_path="./src/main.rs" \
  --var review_depth=thorough
```

---

## Comparison with Existing Tools

| | Nexus | LangChain | AutoGen | CrewAI | Ollama |
|---|---|---|---|---|---|
| Language | Rust | Python | Python | Python | Go |
| Agent process isolation | ✅ Supervised tasks | ❌ | ❌ | ❌ | — |
| Capability enforcement | ✅ Kernel-enforced | ❌ | ❌ | ❌ | — |
| Memory hierarchy | ✅ L1–L4 | ⚠️ Vector only | ⚠️ Flat | ⚠️ Flat | — |
| Tool sandboxing | ✅ WASM + wasmtime | ❌ Direct fn call | ❌ | ❌ | — |
| Fault tolerance | ✅ OTP-style supervision | ❌ | ❌ | ❌ | — |
| Distributed mesh | ✅ libp2p P2P | ❌ | ❌ | ❌ | — |
| Workflow checkpointing | ✅ SQLite-backed | ⚠️ Partial | ❌ | ❌ | — |
| Cost enforcement | ✅ Budget + ledger | ❌ | ❌ | ❌ | — |
| Token streaming | ✅ Backpressure | ⚠️ Polling | ❌ | ❌ | ✅ |
| Session replay | ✅ Full L2 replay | ❌ | ❌ | ❌ | — |
| Live TUI | ✅ ratatui | ❌ | ❌ | ❌ | — |
| GC pauses | ❌ None | ✅ CPython GC | ✅ CPython GC | ✅ CPython GC | ✅ Go GC |

Nexus fills the layer between raw inference and application code. It is the missing systems software of the AI stack.

---

## Project Structure

```
nexus/
├── crates/
│   ├── nexus-proto/       Shared types, traits, protocols (zero deps)
│   ├── nexus-kernel/      Agent process manager, supervisor tree, scheduler
│   ├── nexus-mem/         Four-tier memory hierarchy
│   ├── nexus-tools/       WASM tool sandbox and registry
│   ├── nexus-router/      Universal model gateway
│   ├── nexus-mesh/        Distributed P2P agent fabric
│   ├── nexus-flow/        Workflow DAG engine
│   ├── nexus-obs/         Observability, tracing, TUI, replay
│   └── nexus-cli/         User-facing CLI binary
├── tools/
│   ├── web-search/        Built-in: DuckDuckGo search (WASM)
│   ├── http-fetch/        Built-in: URL fetcher + HTML extractor (WASM)
│   ├── code-exec/         Built-in: Python/JS/Shell/Lua interpreter (WASM)
│   └── file-read/         Built-in: File reader with CSV/JSON/TOML parsing (WASM)
├── examples/
│   ├── research-agent/    Single agent research example
│   ├── multi-agent-pipeline/ Five-agent supervised pipeline
│   └── workflows/         TOML workflow definition examples
├── config/
│   ├── nexus.default.toml Defaults (embedded in binary)
│   └── nexus.example.toml Annotated example for users
└── docs/
    ├── architecture.md
    ├── getting-started.md
    ├── tool-authoring.md
    └── api-reference.md
```

---

## Roadmap

**v0.2.0 — Persistence & Distribution**
- [ ] Agent state persistence across restarts
- [ ] Mesh authentication (noise protocol keypairs)
- [ ] Remote tool execution across mesh nodes
- [ ] WASM tool signing and verification

**v0.3.0 — Developer Experience**
- [ ] `nexus new` project scaffolding
- [ ] Hot-reload for agent code (via dynamic dispatch + versioning)
- [ ] Workflow visual editor (web UI)
- [ ] VS Code extension with inline cost estimates

**v0.4.0 — Production Hardening**
- [ ] Agent memory quotas enforced at the kernel level
- [ ] Distributed tracing across mesh nodes
- [ ] Prometheus metrics endpoint
- [ ] Role-based access control for multi-user deployments

**Future**
- [ ] WASM component model for tool composition
- [ ] Formal agent verification (capability proofs)
- [ ] GPU-accelerated local embedding
- [ ] Nexus Cloud — managed mesh hosting

---

## Contributing

Nexus is MIT-licensed and welcomes contributions. The codebase prioritizes correctness over cleverness, explicitness over magic, and real implementation over stubs.

**Before contributing:**
1. Read [`docs/architecture.md`](docs/architecture.md) — understand the subsystem boundaries
2. Run `cargo clippy --workspace -- -D warnings` — PRs must pass clippy
3. Run `cargo fmt --check` — use `rustfmt` defaults
4. Add tests for any new behavior — we use `#[tokio::test]` throughout

**Good first issues** are labeled [`good-first-issue`](https://github.com/nexus-runtime/nexus/issues?q=label%3Agood-first-issue) on GitHub. The tool ecosystem is the highest-leverage contribution area — if you build a useful WASM tool, open a PR to add it to the built-ins or the community registry.

```bash
# Run the full check suite before opening a PR
cargo check --workspace
cargo clippy --workspace -- -D warnings
cargo fmt --check
cargo test --workspace
cargo doc --workspace --no-deps
```

Join the discussion on [Discord](https://discord.gg/nexus-runtime) or open an issue on [GitHub](https://github.com/nexus-runtime/nexus/issues).

---

## License

Nexus is licensed under the [MIT License](LICENSE).

```
Copyright (c) 2026 Nexus Runtime Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

<div align="center">

**Nexus — To AI agents what `tokio` is to async Rust.**

*Built with ⚡ and a deep appreciation for systems software done right.*

[Documentation](https://docs.rs/nexus-proto) · [Crates.io](https://crates.io/crates/nexus-cli) · [Discord](https://discord.gg/nexus-runtime) · [Issues](https://github.com/nexus-runtime/nexus/issues)

</div>
