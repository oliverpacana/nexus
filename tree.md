nexus/
├── Cargo.toml                          # Workspace manifest
├── Cargo.lock
├── README.md
├── LICENSE
├── .gitignore
├── .github/
│   └── workflows/
│       ├── ci.yml                      # Build, test, clippy, fmt check
│       └── release.yml                 # Publish crates, build binaries
├── config/
│   ├── nexus.default.toml              # Default runtime configuration
│   └── nexus.example.toml             # Annotated example for users
├── docs/
│   ├── architecture.md
│   ├── getting-started.md
│   ├── tool-authoring.md
│   └── api-reference.md
│
├── crates/
│
│   ├── nexus-proto/                    # Shared types, traits, protocols
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  # Re-exports everything
│   │       ├── agent.rs                # AgentId, AgentMeta, AgentState, Priority
│   │       ├── message.rs              # AgentMessage, MessageKind, Envelope
│   │       ├── model.rs                # ModelRequest, ModelResponse, Token, Provider
│   │       ├── tool.rs                 # ToolManifest, ToolCall, ToolResult
│   │       ├── memory.rs              # MemoryKey, MemoryScope, MemoryEntry
│   │       ├── workflow.rs            # WorkflowId, StepId, StepKind, StepStatus
│   │       └── error.rs               # NexusError top-level enum
│
│   ├── nexus-kernel/                   # Agent process manager
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  # KernelHandle, public API
│   │       ├── kernel.rs              # Kernel struct, spawn/kill/list
│   │       ├── agent.rs               # AgentProcess, state machine, task wrapper
│   │       ├── supervisor.rs          # SupervisorTree, RestartStrategy, policies
│   │       ├── scheduler.rs           # Priority queue, rate limiter, token bucket
│   │       ├── capabilities.rs        # CapabilitySet, CapabilityGuard, enforcement
│   │       ├── registry.rs            # DashMap agent registry, lookup
│   │       └── error.rs               # KernelError variants
│
│   ├── nexus-mem/                      # Four-tier memory hierarchy
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  # MemoryStore, unified access API
│   │       ├── working.rs             # L1: Arc<RwLock> in-process working memory
│   │       ├── episodic.rs            # L2: SQLite event log via libsql
│   │       ├── semantic.rs            # L3: Vector index via usearch + embeddings
│   │       ├── procedural.rs          # L4: Knowledge graph via sled
│   │       ├── permissions.rs         # Scope enforcement, grant table
│   │       ├── embeddings.rs          # Embedding model abstraction + local impl
│   │       └── error.rs               # MemoryError variants
│
│   ├── nexus-tools/                    # WASM plugin sandbox
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  # ToolEngine, public API
│   │       ├── engine.rs              # wasmtime engine setup, linker config
│   │       ├── sandbox.rs             # WasmSandbox, per-call isolation
│   │       ├── manifest.rs            # ToolManifest parsing, validation
│   │       ├── loader.rs              # Load .wasm from disk/registry
│   │       ├── registry.rs            # In-memory tool registry, versioning
│   │       ├── host_functions.rs      # WASM host imports (http, log, clock)
│   │       └── error.rs               # ToolError variants
│
│   ├── nexus-router/                   # Universal model gateway
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  # ModelRouter, public API
│   │       ├── router.rs              # Routing logic, policy evaluation
│   │       ├── policy.rs              # RoutingPolicy enum, evaluator
│   │       ├── stream.rs              # Token streaming, backpressure, framing
│   │       ├── cost.rs                # Cost estimation, per-model pricing table
│   │       ├── providers/
│   │       │   ├── mod.rs             # ModelProvider trait definition
│   │       │   ├── openai.rs          # OpenAI & compatible (GPT-4o, o1, etc.)
│   │       │   ├── anthropic.rs       # Anthropic (Claude 3.x, Claude 4.x)
│   │       │   ├── groq.rs            # Groq (Llama, Mixtral, Gemma)
│   │       │   ├── mistral.rs         # Mistral AI provider
│   │       │   └── local.rs           # Local llama.cpp via HTTP or FFI
│   │       └── error.rs               # RouterError variants
│
│   ├── nexus-mesh/                     # Distributed P2P agent fabric
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  # MeshNode, public API
│   │       ├── node.rs                # MeshNode struct, peer management
│   │       ├── network.rs             # libp2p setup, transport, behaviour
│   │       ├── discovery.rs           # mDNS + rendezvous server discovery
│   │       ├── routing.rs             # Capability-based work routing
│   │       ├── blackboard.rs          # CRDT distributed shared memory
│   │       ├── crdt.rs                # CRDT types: LWW-Map, G-Set, OR-Set
│   │       ├── protocol.rs            # Wire protocol, message framing
│   │       └── error.rs               # MeshError variants
│
│   ├── nexus-flow/                     # Workflow DAG engine
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  # WorkflowEngine, public API
│   │       ├── dag.rs                 # DAG struct, topological sort, cycle detection
│   │       ├── executor.rs            # DAG execution, parallel branches, join
│   │       ├── step.rs                # Step types, StepContext, StepResult
│   │       ├── checkpoint.rs          # Checkpoint serialization, SQLite store
│   │       ├── dsl.rs                 # Rust builder DSL for workflow definition
│   │       ├── loader.rs              # TOML/YAML workflow file parser
│   │       ├── condition.rs           # Structured output conditions, JSON schema routing
│   │       └── error.rs               # FlowError variants
│
│   ├── nexus-obs/                      # Observability, tracing, replay
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  # ObsHandle, public API
│   │       ├── tracer.rs              # tokio-tracing integration, span management
│   │       ├── exporter.rs            # OpenTelemetry export (OTLP, Jaeger)
│   │       ├── ledger.rs              # Cost ledger: per-agent token accounting
│   │       ├── replay.rs              # Replay engine: rebuild agent run from L2 mem
│   │       ├── tui/
│   │       │   ├── mod.rs             # TUI entry point, app state
│   │       │   ├── app.rs             # TuiApp struct, event loop
│   │       │   ├── widgets/
│   │       │   │   ├── mesh_tree.rs   # Live agent mesh tree widget
│   │       │   │   ├── log_panel.rs   # Scrollable log panel
│   │       │   │   ├── cost_table.rs  # Cost breakdown table
│   │       │   │   └── dag_view.rs    # Current workflow DAG visualizer
│   │       │   └── theme.rs           # Color theme, styles
│   │       └── error.rs               # ObsError variants
│
│   └── nexus-cli/                      # User-facing CLI binary
│       ├── Cargo.toml
│       └── src/
│           ├── main.rs                 # CLI entry point, clap app setup
│           ├── config.rs              # Config loading, env merging
│           ├── runtime.rs             # Runtime bootstrap (start all subsystems)
│           └── commands/
│               ├── mod.rs             # Command dispatch
│               ├── up.rs              # `nexus up`: start the runtime daemon
│               ├── run.rs             # `nexus run <workflow>`: execute workflow
│               ├── status.rs          # `nexus status`: show running agents
│               ├── tui.rs             # `nexus tui`: launch live dashboard
│               ├── mem.rs             # `nexus mem inspect/search/clear`
│               └── tool.rs            # `nexus tool install/list/remove`
│
├── tools/                              # First-party built-in tool plugins (WASM)
│   ├── web-search/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── lib.rs                  # WASM: query → search results JSON
│   ├── code-exec/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── lib.rs                  # WASM: code string → execution result
│   ├── http-fetch/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── lib.rs                  # WASM: URL → page content
│   └── file-read/
│       ├── Cargo.toml
│       └── src/
│           └── lib.rs                  # WASM: path → file content (sandboxed)
│
└── examples/
    ├── research-agent/
    │   ├── Cargo.toml
    │   └── src/
    │       └── main.rs                 # Single agent research example
    ├── multi-agent-pipeline/
    │   ├── Cargo.toml
    │   └── src/
    │       └── main.rs                 # Research + analyze + write pipeline
    └── workflows/
        ├── research-and-write.toml    # Example workflow definition
        └── code-review.toml           # Example workflow definition
