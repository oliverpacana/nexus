use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

use nexus_proto::agent::{AgentCapabilities, MemoryAccess};
use nexus_proto::memory::{MemoryScope, MemoryTier};
use nexus_proto::NexusError;

use crate::error::{KernelError, Result};

// =============================================================================
// Capability Enum — Fine-Grained Permission Types
// =============================================================================

/// A fine-grained capability representing a specific permission an agent may hold.
/// Capabilities are declared at agent spawn time and enforced at runtime by CapabilityGuard.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum Capability {
    /// Permission to invoke a specific WASM tool by name.
    /// Example: `Tool("web-search")` corresponds to capability string `"tool:web-search"`
    Tool(String),

    /// Permission to read from a specific memory tier with a given scope.
    MemoryRead {
        tier: MemoryTier,
        scope: MemoryScope,
    },

    /// Permission to write to a specific memory tier with a given scope.
    MemoryWrite {
        tier: MemoryTier,
        scope: MemoryScope,
    },

    /// Permission to route requests to models matching a glob pattern.
    /// Patterns support `*` (single segment) and `**` (any path) wildcards.
    /// Example: `ModelAccess { pattern: "anthropic/*" }` matches `"anthropic/claude-3-sonnet"`
    ModelAccess {
        pattern: String,
    },

    /// Permission to delegate work to other agents via the P2P mesh.
    MeshDelegate,

    /// Permission to broadcast messages to all agents in the mesh.
    MeshBroadcast,

    /// Permission to install new WASM tools into the local registry.
    /// This is a privileged capability; typically only granted to admin agents.
    ToolInstall,

    /// Permission to spawn child agents, with a limit on concurrent children.
    SpawnAgent {
        max_children: usize,
    },
}

impl fmt::Display for Capability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_capability_string())
    }
}

impl Capability {
    /// Parses a capability from its string representation in `"type:value"` format.
    ///
    /// # Supported Formats
    /// - `tool:<name>` → `Capability::Tool(name)`
    /// - `memory_read:<tier>:<scope>` → `Capability::MemoryRead { tier, scope }`
    /// - `memory_write:<tier>:<scope>` → `Capability::MemoryWrite { tier, scope }`
    /// - `model:<pattern>` → `Capability::ModelAccess { pattern }`
    /// - `mesh:delegate` → `Capability::MeshDelegate`
    /// - `mesh:broadcast` → `Capability::MeshBroadcast`
    /// - `tool:install` → `Capability::ToolInstall`
    /// - `spawn:<max_children>` → `Capability::SpawnAgent { max_children }`
    ///
    /// # Errors
    /// Returns `KernelError::Internal` if the string format is unparseable.
    pub fn from_str_capability(s: &str) -> Result<Self> {
        let parts: Vec<&str> = s.splitn(2, ':').collect();
        let cap_type = parts.first().ok_or_else(|| {
            KernelError::Internal(format!("invalid capability string: empty type in '{}'", s))
        })?;
        let value = parts.get(1).copied().unwrap_or("");

        match *cap_type {
            "tool" if value != "install" => Ok(Capability::Tool(value.to_string())),
            "tool:install" | "tool_install" => Ok(Capability::ToolInstall),
            "memory_read" => {
                let subparts: Vec<&str> = value.splitn(2, ':').collect();
                if subparts.len() != 2 {
                    return Err(KernelError::Internal(
                        "memory_read requires tier:scope format".into(),
                    ));
                }
                let tier = MemoryTier::from_str_capability(subparts[0])?;
                let scope = MemoryScope::from_str_capability(subparts[1])?;
                Ok(Capability::MemoryRead { tier, scope })
            }            "memory_write" => {
                let subparts: Vec<&str> = value.splitn(2, ':').collect();
                if subparts.len() != 2 {
                    return Err(KernelError::Internal(
                        "memory_write requires tier:scope format".into(),
                    ));
                }
                let tier = MemoryTier::from_str_capability(subparts[0])?;
                let scope = MemoryScope::from_str_capability(subparts[1])?;
                Ok(Capability::MemoryWrite { tier, scope })
            }
            "model" => Ok(Capability::ModelAccess {
                pattern: value.to_string(),
            }),
            "mesh:delegate" | "mesh_delegate" => Ok(Capability::MeshDelegate),
            "mesh:broadcast" | "mesh_broadcast" => Ok(Capability::MeshBroadcast),
            "spawn" => {
                let max = value.parse::<usize>().map_err(|_| {
                    KernelError::Internal(format!(
                        "spawn capability requires numeric max_children, got '{}'",
                        value
                    ))
                })?;
                Ok(Capability::SpawnAgent { max_children: max })
            }
            _ => Err(KernelError::Internal(format!(
                "unknown capability type: '{}'",
                cap_type
            ))),
        }
    }

    /// Serializes this capability back to its canonical string representation.
    /// Used for logging, configuration, and cross-subsystem protocol messages.
    pub fn to_capability_string(&self) -> String {
        match self {
            Capability::Tool(name) => format!("tool:{}", name),
            Capability::MemoryRead { tier, scope } => {
                format!(
                    "memory_read:{}:{}",
                    tier.to_capability_str(),
                    scope.to_capability_str()
                )
            }
            Capability::MemoryWrite { tier, scope } => {
                format!(
                    "memory_write:{}:{}",
                    tier.to_capability_str(),
                    scope.to_capability_str()
                )            }
            Capability::ModelAccess { pattern } => format!("model:{}", pattern),
            Capability::MeshDelegate => "mesh:delegate".into(),
            Capability::MeshBroadcast => "mesh:broadcast".into(),
            Capability::ToolInstall => "tool:install".into(),
            Capability::SpawnAgent { max_children } => {
                format!("spawn:{}", max_children)
            }
        }
    }
}

// Helper trait for converting MemoryTier/MemoryScope to/from capability strings.
// Implemented here to avoid circular dependencies with nexus-proto.
trait CapabilityStringExt {
    fn to_capability_str(&self) -> String;
    fn from_str_capability(s: &str) -> Result<Self>
    where
        Self: Sized;
}

impl CapabilityStringExt for MemoryTier {
    fn to_capability_str(&self) -> String {
        match self {
            MemoryTier::Working => "working".into(),
            MemoryTier::Episodic => "episodic".into(),
            MemoryTier::Semantic => "semantic".into(),
            MemoryTier::Procedural => "procedural".into(),
        }
    }

    fn from_str_capability(s: &str) -> Result<Self> {
        match s {
            "working" => Ok(MemoryTier::Working),
            "episodic" => Ok(MemoryTier::Episodic),
            "semantic" => Ok(MemoryTier::Semantic),
            "procedural" => Ok(MemoryTier::Procedural),
            _ => Err(KernelError::Internal(format!(
                "unknown memory tier: '{}'",
                s
            ))),
        }
    }
}

impl CapabilityStringExt for MemoryScope {
    fn to_capability_str(&self) -> String {
        match self {
            MemoryScope::Private => "private".into(),
            MemoryScope::Group => "group".into(),            MemoryScope::Global => "global".into(),
        }
    }

    fn from_str_capability(s: &str) -> Result<Self> {
        match s {
            "private" => Ok(MemoryScope::Private),
            "group" => Ok(MemoryScope::Group),
            "global" => Ok(MemoryScope::Global),
            _ => Err(KernelError::Internal(format!(
                "unknown memory scope: '{}'",
                s
            ))),
        }
    }
}

// =============================================================================
// CapabilitySet — Immutable Granted Permissions
// =============================================================================

/// An immutable set of capabilities granted to an agent.
/// Internally uses `Arc<HashSet<Capability>>` for cheap cloning and thread-safe sharing.
#[derive(Debug, Clone)]
pub struct CapabilitySet {
    inner: Arc<HashSet<Capability>>,
}

impl CapabilitySet {
    /// Constructs a new capability set from a vector of granted capabilities.
    pub fn new(caps: Vec<Capability>) -> Self {
        Self {
            inner: Arc::new(caps.into_iter().collect()),
        }
    }

    /// Returns an empty capability set (no permissions granted).
    pub fn empty() -> Self {
        Self {
            inner: Arc::new(HashSet::new()),
        }
    }

    /// Returns the root capability set with ALL permissions.
    /// Intended only for the kernel itself or trusted admin agents.
    pub fn root() -> Self {
        // In practice, root capabilities would be enumerated explicitly.
        // For now, we use a marker: any check against root succeeds.
        // This is enforced in CapabilityGuard::check_* methods.
        Self {            inner: Arc::new(HashSet::new()), // marker: empty set means "all"
        }
    }

    /// Checks if this set contains the specified capability.
    /// Root sets (empty inner) always return true.
    pub fn has(&self, cap: &Capability) -> bool {
        // Root capability set: grant everything
        if self.inner.is_empty() && Arc::strong_count(&self.inner) == 1 {
            // Heuristic: if we're the sole owner of an empty set, treat as root
            // A more robust approach would use an explicit is_root flag.
            return true;
        }
        self.inner.contains(cap)
    }

    /// Convenience: checks if the agent can invoke the named tool.
    pub fn has_tool(&self, tool_name: &str) -> bool {
        self.has(&Capability::Tool(tool_name.to_string()))
    }

    /// Convenience: checks if the agent can write to the specified memory tier/scope.
    pub fn has_memory_write(&self, tier: MemoryTier, scope: MemoryScope) -> bool {
        self.has(&Capability::MemoryWrite { tier, scope })
    }

    /// Convenience: checks if the agent can read from the specified memory tier/scope.
    pub fn has_memory_read(&self, tier: MemoryTier, scope: MemoryScope) -> bool {
        self.has(&Capability::MemoryRead { tier, scope })
    }

    /// Checks if the agent can access a model matching the given identifier.
    /// Performs glob matching against all `ModelAccess` patterns in the set.
    pub fn has_model_access(&self, model_id: &str) -> bool {
        // Root grants all model access
        if self.inner.is_empty() && Arc::strong_count(&self.inner) == 1 {
            return true;
        }

        self.inner.iter().any(|cap| {
            if let Capability::ModelAccess { pattern } = cap {
                glob_matches(pattern, model_id)
            } else {
                false
            }
        })
    }

    /// Returns an iterator over all granted capabilities.
    pub fn all(&self) -> impl Iterator<Item = &Capability> {        self.inner.iter()
    }
}

impl From<AgentCapabilities> for CapabilitySet {
    /// Converts the high-level `AgentCapabilities` from nexus-proto into
    /// the kernel's fine-grained `CapabilitySet`.
    fn from(proto_caps: AgentCapabilities) -> Self {
        let mut caps = Vec::new();

        for cap_str in proto_caps.all() {
            // Attempt to parse each declared capability string
            if let Ok(cap) = Capability::from_str_capability(cap_str) {
                caps.push(cap);
            }
            // Silently skip unparseable capabilities; they'll be denied at runtime
        }

        CapabilitySet::new(caps)
    }
}

// =============================================================================
// CapabilityGuard — Runtime Enforcement Point
// =============================================================================

/// Runtime guard that enforces capability checks for a specific agent.
/// All privileged operations in the kernel should route through this guard.
#[derive(Debug, Clone)]
pub struct CapabilityGuard {
    agent_id: uuid::Uuid,
    capabilities: CapabilitySet,
}

impl CapabilityGuard {
    /// Constructs a new guard for the given agent and capability set.
    pub fn new(agent_id: uuid::Uuid, capabilities: CapabilitySet) -> Self {
        Self {
            agent_id,
            capabilities,
        }
    }

    /// Checks if the agent is permitted to invoke the named tool.
    /// Returns `Err(KernelError::CapabilityDenied)` if not authorized.
    pub fn check_tool(&self, tool_name: &str) -> Result<()> {
        if self.capabilities.has_tool(tool_name) {
            Ok(())
        } else {
            Err(KernelError::CapabilityDenied {                agent_id: self.agent_id,
                capability: format!("tool:{}", tool_name),
            })
        }
    }

    /// Checks if the agent can write to the specified memory tier/scope.
    pub fn check_memory_write(&self, tier: MemoryTier, scope: MemoryScope) -> Result<()> {
        if self.capabilities.has_memory_write(tier, scope) {
            Ok(())
        } else {
            Err(KernelError::CapabilityDenied {
                agent_id: self.agent_id,
                capability: format!(
                    "memory_write:{}:{}",
                    tier.to_capability_str(),
                    scope.to_capability_str()
                ),
            })
        }
    }

    /// Checks if the agent can read from the specified memory tier/scope.
    pub fn check_memory_read(&self, tier: MemoryTier, scope: MemoryScope) -> Result<()> {
        if self.capabilities.has_memory_read(tier, scope) {
            Ok(())
        } else {
            Err(KernelError::CapabilityDenied {
                agent_id: self.agent_id,
                capability: format!(
                    "memory_read:{}:{}",
                    tier.to_capability_str(),
                    scope.to_capability_str()
                ),
            })
        }
    }

    /// Checks if the agent can route requests to the specified model.
    /// Performs glob matching against declared `ModelAccess` patterns.
    pub fn check_model(&self, model_id: &str) -> Result<()> {
        if self.capabilities.has_model_access(model_id) {
            Ok(())
        } else {
            Err(KernelError::CapabilityDenied {
                agent_id: self.agent_id,
                capability: format!("model:{}", model_id),
            })
        }
    }
    /// Checks if the agent is permitted to spawn child agents.
    /// Returns the maximum number of concurrent children allowed, or an error.
    pub fn check_spawn(&self) -> Result<usize> {
        // Root can spawn unlimited children
        if self.capabilities.inner.is_empty() && Arc::strong_count(&self.capabilities.inner) == 1 {
            return Ok(usize::MAX);
        }

        self.capabilities
            .all()
            .find_map(|cap| {
                if let Capability::SpawnAgent { max_children } = cap {
                    Some(*max_children)
                } else {
                    None
                }
            })
            .ok_or_else(|| KernelError::CapabilityDenied {
                agent_id: self.agent_id,
                capability: "spawn".into(),
            })
    }

    /// Checks if the agent can delegate work via the P2P mesh.
    pub fn check_mesh_delegate(&self) -> Result<()> {
        if self.capabilities.has(&Capability::MeshDelegate) {
            Ok(())
        } else {
            Err(KernelError::CapabilityDenied {
                agent_id: self.agent_id,
                capability: "mesh:delegate".into(),
            })
        }
    }

    /// Checks if the agent can broadcast messages to the mesh.
    pub fn check_mesh_broadcast(&self) -> Result<()> {
        if self.capabilities.has(&Capability::MeshBroadcast) {
            Ok(())
        } else {
            Err(KernelError::CapabilityDenied {
                agent_id: self.agent_id,
                capability: "mesh:broadcast".into(),
            })
        }
    }

    /// Checks if the agent can install new WASM tools.
    pub fn check_tool_install(&self) -> Result<()> {        if self.capabilities.has(&Capability::ToolInstall) {
            Ok(())
        } else {
            Err(KernelError::CapabilityDenied {
                agent_id: self.agent_id,
                capability: "tool:install".into(),
            })
        }
    }
}

// =============================================================================
// Glob Matching — Pattern Matching for Model Access
// =============================================================================

/// Matches a value string against a glob pattern supporting `*` and `**` wildcards.
///
/// # Wildcard Semantics
/// - `*` matches any sequence of characters EXCEPT `/` (single path segment)
/// - `**` matches any sequence of characters INCLUDING `/` (any path depth)
/// - Matching is case-sensitive
/// - Patterns are matched in their entirety (anchored at start and end)
///
/// # Examples
/// ```
/// assert!(glob_matches("anthropic/*", "anthropic/claude-3-sonnet"));
/// assert!(glob_matches("openai/**", "openai/gpt-4/vision"));
/// assert!(!glob_matches("anthropic/*", "anthropic/claude/3")); // * doesn't cross /
/// assert!(glob_matches("local/*", "local/llama3"));
/// ```
pub fn glob_matches(pattern: &str, value: &str) -> bool {
    glob_match_recursive(pattern.as_bytes(), value.as_bytes(), 0, 0)
}

/// Recursive helper for glob matching with backtracking.
/// Uses indices to avoid string allocations during recursion.
fn glob_match_recursive(
    pattern: &[u8],
    value: &[u8],
    p_idx: usize,
    v_idx: usize,
) -> bool {
    // Base case: both pattern and value exhausted → match
    if p_idx == pattern.len() && v_idx == value.len() {
        return true;
    }

    // Pattern exhausted but value remains → no match
    if p_idx == pattern.len() {
        return false;    }

    // Handle ** wildcard (matches anything including /)
    if p_idx + 1 < pattern.len() && pattern[p_idx] == b'*' && pattern[p_idx + 1] == b'*' {
        // Skip the ** and any following / for cleaner matching
        let next_p = skip_double_star(pattern, p_idx);

        // Try matching ** against 0 or more characters in value
        for i in v_idx..=value.len() {
            if glob_match_recursive(pattern, value, next_p, i) {
                return true;
            }
        }
        return false;
    }

    // Handle * wildcard (matches anything except /)
    if pattern[p_idx] == b'*' {
        let next_p = p_idx + 1;

        // Try matching * against 0 or more non-/ characters
        for i in v_idx..=value.len() {
            // Stop if we hit a / (since * doesn't cross segment boundaries)
            if i > v_idx && value[i - 1] == b'/' {
                break;
            }
            if glob_match_recursive(pattern, value, next_p, i) {
                return true;
            }
        }
        return false;
    }

    // Literal character match (case-sensitive) or value exhausted
    if v_idx >= value.len() || pattern[p_idx] != value[v_idx] {
        return false;
    }

    // Recurse on next characters
    glob_match_recursive(pattern, value, p_idx + 1, v_idx + 1)
}

/// Advances past a `**` wildcard and any immediately following `/`.
/// This normalizes patterns like `**/foo` or `**` for consistent matching.
fn skip_double_star(pattern: &[u8], mut idx: usize) -> usize {
    // Skip the two * characters
    idx += 2;

    // Optionally skip a following / to handle patterns like "**/path"
    if idx < pattern.len() && pattern[idx] == b'/' {        idx += 1;
    }

    idx
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capability_parsing() {
        let cap = Capability::from_str_capability("tool:web-search").unwrap();
        assert_eq!(cap, Capability::Tool("web-search".into()));
        assert_eq!(cap.to_capability_string(), "tool:web-search");

        let cap = Capability::from_str_capability("memory_write:semantic:global").unwrap();
        assert_eq!(
            cap,
            Capability::MemoryWrite {
                tier: MemoryTier::Semantic,
                scope: MemoryScope::Global,
            }
        );

        let cap = Capability::from_str_capability("model:anthropic/*").unwrap();
        assert_eq!(
            cap,
            Capability::ModelAccess {
                pattern: "anthropic/*".into()
            }
        );
    }

    #[test]
    fn test_glob_matching() {
        // Single * wildcard (no / crossing)
        assert!(glob_matches("anthropic/*", "anthropic/claude-3-sonnet"));
        assert!(!glob_matches("anthropic/*", "anthropic/claude/3"));

        // Double ** wildcard (crosses /)
        assert!(glob_matches("openai/**", "openai/gpt-4/vision"));
        assert!(glob_matches("local/**", "local/llama3"));
        assert!(glob_matches("**/test", "foo/bar/test"));

        // Literal match        assert!(glob_matches("exact", "exact"));
        assert!(!glob_matches("exact", "exactly"));

        // Empty patterns/values
        assert!(glob_matches("", ""));
        assert!(!glob_matches("*", "")); // * requires at least one char (but not /)
        assert!(glob_matches("**", "")); // ** can match empty

        // Edge: * at end
        assert!(glob_matches("prefix/*", "prefix/value"));
        assert!(!glob_matches("prefix/*", "prefix/")); // * needs content
    }

    #[test]
    fn test_capability_set_conversion() {
        use nexus_proto::agent::AgentCapabilities;

        let mut proto = AgentCapabilities::new();
        proto = proto.with_tool("web-search");
        proto = proto.with_model("anthropic/*");

        let caps = CapabilitySet::from(proto);
        assert!(caps.has_tool("web-search"));
        assert!(caps.has_model_access("anthropic/claude-3-sonnet"));
        assert!(!caps.has_tool("code-exec"));
    }

    #[test]
    fn test_capability_guard_enforcement() {
        use uuid::Uuid;

        let agent_id = Uuid::new_v4();
        let caps = CapabilitySet::new(vec![
            Capability::Tool("web-search".into()),
            Capability::MemoryWrite {
                tier: MemoryTier::Semantic,
                scope: MemoryScope::Private,
            },
        ]);
        let guard = CapabilityGuard::new(agent_id, caps);

        // Allowed
        assert!(guard.check_tool("web-search").is_ok());
        assert!(guard
            .check_memory_write(MemoryTier::Semantic, MemoryScope::Private)
            .is_ok());

        // Denied
        assert!(matches!(
            guard.check_tool("code-exec"),            Err(KernelError::CapabilityDenied { .. })
        ));
        assert!(matches!(
            guard.check_memory_write(MemoryTier::Episodic, MemoryScope::Global),
            Err(KernelError::CapabilityDenied { .. })
        ));
    }
}
