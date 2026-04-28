use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use dashmap::DashMap;
use nexus_proto::tool::{ToolCall, ToolManifest as ProtoManifest, ToolId, ToolResult};
use serde::{Deserialize, Serialize};
use tokio::task::spawn_blocking;
use tracing::{debug, error, info, instrument, warn};
use uuid::Uuid;
use wasmtime::Engine;

use crate::error::{ToolError, Result};
use crate::manifest::{load_manifest_alongside_wasm, load_manifest_from_file};
use crate::sandbox::{create_shared_engine, WasmSandbox};

// =============================================================================
// ToolVersion — Semver-like Versioning
// =============================================================================

/// A semantic version struct for tool versioning and comparison.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ToolVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl ToolVersion {
    /// Parses a version string in `"major.minor.patch"` format.
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim().trim_start_matches('v');
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 3 {
            return Err(ToolError::ManifestInvalid {
                field: "version".into(),
                reason: "must be in major.minor.patch format".into(),
            });
        }

        Ok(Self {
            major: parts[0].parse().map_err(|_| ToolError::ManifestInvalid {
                field: "version.major".into(),
                reason: "must be numeric".into(),
            })?,
            minor: parts[1].parse().map_err(|_| ToolError::ManifestInvalid {                field: "version.minor".into(),
                reason: "must be numeric".into(),
            })?,
            patch: parts[2].parse().map_err(|_| ToolError::ManifestInvalid {
                field: "version.patch".into(),
                reason: "must be numeric".into(),
            })?,
        })
    }

    /// Returns `true` if this version is strictly newer than `other`.
    pub fn is_newer_than(&self, other: &Self) -> bool {
        self > other
    }
}

impl fmt::Display for ToolVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl PartialOrd for ToolVersion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ToolVersion {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.major
            .cmp(&other.major)
            .then_with(|| self.minor.cmp(&other.minor))
            .then_with(|| self.patch.cmp(&other.patch))
    }
}

impl Default for ToolVersion {
    fn default() -> Self {
        Self {
            major: 0,
            minor: 0,
            patch: 0,
        }
    }
}

// =============================================================================
// ToolRegistration & ToolStats
// =============================================================================
/// Runtime registration of an installed tool, including metrics.
pub struct ToolRegistration {
    /// Parsed manifest metadata.
    pub manifest: ProtoManifest,

    /// Pre-compiled WASM sandbox (thread-safe, reusable).
    pub sandbox: Arc<WasmSandbox>,

    /// Timestamp when this version was installed.
    pub installed_at: DateTime<Utc>,

    /// Atomic counter: total invocations.
    pub call_count: AtomicU64,

    /// Atomic counter: failed invocations.
    pub error_count: AtomicU64,

    /// Atomic counter: cumulative wall-clock execution time in ms.
    pub total_execution_ms: AtomicU64,
}

impl ToolRegistration {
    /// Records a completed tool invocation for metrics tracking.
    pub fn record_call(&self, duration_ms: u64, success: bool) {
        self.call_count.fetch_add(1, Ordering::Relaxed);
        self.total_execution_ms.fetch_add(duration_ms, Ordering::Relaxed);
        if !success {
            self.error_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Returns a snapshot of usage statistics for this tool version.
    pub fn stats(&self) -> ToolStats {
        let calls = self.call_count.load(Ordering::Relaxed);
        let errors = self.error_count.load(Ordering::Relaxed);
        let total_ms = self.total_execution_ms.load(Ordering::Relaxed);

        let version_str = ToolVersion::parse(&self.manifest.version)
            .map(|v| v.to_string())
            .unwrap_or_else(|_| self.manifest.version.clone());

        ToolStats {
            name: self.manifest.name.clone(),
            version: version_str,
            call_count: calls,
            error_count: errors,
            avg_execution_ms: if calls > 0 {
                total_ms as f64 / calls as f64
            } else {                0.0
            },
            success_rate: if calls > 0 {
                (calls.saturating_sub(errors)) as f64 / calls as f64
            } else {
                1.0
            },
        }
    }
}

/// Snapshot of tool usage metrics for observability and CLI output.
#[derive(Debug, Clone)]
pub struct ToolStats {
    pub name: String,
    pub version: String,
    pub call_count: u64,
    pub error_count: u64,
    pub avg_execution_ms: f64,
    pub success_rate: f64,
}

// =============================================================================
// ToolRegistry — Central Tool Index
// =============================================================================

/// The central registry for installed WASM tools.
///
/// # Design
/// - Tools are indexed by `name → version` using nested `DashMap` for lock-free concurrent access
/// - Latest version tracking enables `get(name, None)` to resolve automatically
/// - Hot-reload support allows replacing tool implementations at runtime
/// - All installations include checksum verification and schema validation
///
/// # Thread Safety
/// - `Send + Sync`; safe for concurrent access from many Tokio tasks
/// - `DashMap` ensures atomic check-and-insert for concurrent installations
/// - Metrics use `AtomicU64` for wait-free counters
pub struct ToolRegistry {
    /// Nested index: tool name → version → registration
    tools: DashMap<String, DashMap<ToolVersion, Arc<ToolRegistration>>>,

    /// Tracks the highest installed version for each tool name
    latest: DashMap<String, ToolVersion>,

    /// Root directory for installed tool artifacts
    registry_path: PathBuf,

    /// Shared wasmtime engine (configured for fuel/memory limits)
    engine: Arc<Engine>,}

impl ToolRegistry {
    /// Creates a new tool registry, initializing the wasmtime engine and registry directory.
    #[instrument(skip(registry_path), fields(path = ?registry_path))]
    pub fn new(registry_path: PathBuf) -> Result<Self> {
        debug!("initializing tool registry");

        // Create registry directory if it doesn't exist
        fs::create_dir_all(&registry_path).map_err(|e| ToolError::IoError(e))?;

        // Create shared engine with sandbox defaults
        let engine = create_shared_engine()?;

        Ok(Self {
            tools: DashMap::new(),
            latest: DashMap::new(),
            registry_path,
            engine,
        })
    }

    /// Installs a new tool version into the registry.
    ///
    /// # Flow
    /// 1. Loads manifest (adjacent to WASM or from explicit path)
    /// 2. Verifies SHA-256 checksum if specified in manifest
    /// 3. Copies artifacts to `{registry_path}/{name}/{version}/`
    /// 4. Compiles WASM module via `WasmSandbox::new`
    /// 5. Registers in concurrent index
    ///
    /// # Arguments
    /// * `wasm_path` - Path to the compiled `.wasm` binary
    /// * `manifest_path` - Optional explicit path to `manifest.toml`
    ///
    /// # Returns
    /// * `Ok(ToolId)` - The installed tool identifier
    /// * `Err(ToolError)` - If validation, checksum, or compilation fails
    #[instrument(skip(self, wasm_path), fields(path = ?wasm_path))]
    pub async fn install(
        &self,
        wasm_path: &Path,
        manifest_path: Option<&Path>,
    ) -> Result<ToolId> {
        debug!("installing tool");

        // 1. Load manifest
        let manifest = if let Some(path) = manifest_path {
            load_manifest_from_file(path)?
        } else {            load_manifest_alongside_wasm(wasm_path)?
        };

        let tool_name = manifest.name.clone();
        let version = ToolVersion::parse(&manifest.version)?;
        let tool_dir = self.registry_path.join(&tool_name).join(version.to_string());

        // 2. Verify checksum if present
        if let Some(expected) = &manifest.checksum_sha256 {
            let wasm_bytes = fs::read(wasm_path).map_err(ToolError::IoError)?;
            let actual = compute_wasm_checksum(&wasm_bytes);
            if !actual.eq_ignore_ascii_case(expected) {
                return Err(ToolError::ChecksumMismatch {
                    expected: expected.clone(),
                    actual,
                });
            }
        }

        // 3. Copy artifacts to registry directory
        fs::create_dir_all(&tool_dir).map_err(ToolError::IoError)?;
        let dest_wasm = tool_dir.join("tool.wasm");
        let dest_manifest = tool_dir.join("manifest.toml");

        fs::copy(wasm_path, &dest_wasm).map_err(ToolError::IoError)?;

        // If manifest wasn't explicitly provided, copy the parsed TOML
        if manifest_path.is_none() {
            // Re-read original to preserve formatting
            if let Ok(original_toml) = fs::read_to_string(
                manifest_path.unwrap_or_else(|| {
                    wasm_path.parent().map_or(Path::new(""), |p| {
                        p.join(&tool_name).with_extension("toml")
                    })
                }),
            ) {
                fs::write(&dest_manifest, original_toml).ok();
            }
        } else if let Some(path) = manifest_path {
            fs::copy(path, &dest_manifest).ok();
        }

        // 4. Compile WASM (CPU-intensive, run on blocking thread)
        let wasm_bytes = fs::read(&dest_wasm).map_err(ToolError::IoError)?;
        let engine = Arc::clone(&self.engine);
        let manifest_clone = manifest.clone();

        let sandbox = spawn_blocking(move || {
            WasmSandbox::new(engine, manifest_clone, wasm_bytes)
        })        .await
        .map_err(|e| ToolError::Internal(format!("compilation panicked: {}", e)))??;

        // 5. Create registration
        let registration = Arc::new(ToolRegistration {
            manifest: manifest.clone(),
            sandbox: Arc::new(sandbox),
            installed_at: Utc::now(),
            call_count: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
            total_execution_ms: AtomicU64::new(0),
        });

        // 6. Register atomically
        let versions = self.tools.entry(tool_name.clone()).or_default();
        versions.insert(version.clone(), registration);

        // Update latest version tracking
        let update_latest = self.latest.get(&tool_name).map_or(true, |entry| {
            version > *entry
        });
        if update_latest {
            self.latest.insert(tool_name, version);
        }

        info!(
            name = %manifest.name,
            version = %version,
            "tool installed successfully"
        );

        Ok(ToolId::new(&manifest.name, &manifest.version))
    }

    /// Retrieves a registered tool by name and optional version.
    ///
    /// If `version` is `None`, resolves to the latest installed version.
    pub fn get(&self, name: &str, version: Option<&str>) -> Option<Arc<ToolRegistration>> {
        let versions = self.tools.get(name)?;

        let target_version = if let Some(v) = version {
            ToolVersion::parse(v).ok()?
        } else {
            self.latest.get(name).map(|e| e.value().clone())?
        };

        versions.get(&target_version).map(|e| e.value().clone())
    }

    /// Returns a list of usage statistics for all installed tools.    /// Returns one entry per installed version.
    pub fn list(&self) -> Vec<ToolStats> {
        self.tools
            .iter()
            .flat_map(|entry| {
                entry
                    .value()
                    .iter()
                    .map(|v| v.value().stats())
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Uninstalls a tool version from the registry.
    ///
    /// If `version` is `None`, removes all versions of the named tool.
    /// Removes from memory and deletes artifacts from disk.
    #[instrument(skip(self), fields(name))]
    pub async fn uninstall(&self, name: &str, version: Option<&str>) -> Result<()> {
        debug!("uninstalling tool");

        let tool_dir = self.registry_path.join(name);
        if !tool_dir.exists() {
            return Err(ToolError::NotFound(tool_dir));
        }

        if let Some(v) = version {
            let version = ToolVersion::parse(v)?;
            let versions = self.tools.get(name).ok_or_else(|| {
                ToolError::NotFound(tool_dir.clone())
            })?;

            if versions.remove(&version).is_none() {
                return Err(ToolError::NotFound(tool_dir.join(v)));
            }

            // Update latest if needed
            if self
                .latest
                .get(name)
                .map_or(false, |e| *e == version)
            {
                self.latest.remove(name);
                // Recompute latest
                if let Some(versions) = self.tools.get(name) {
                    versions
                        .iter()
                        .max_by_key(|e| e.key().clone())
                        .map(|e| self.latest.insert(name.to_string(), e.key().clone()));                }
            }

            // Delete directory
            let version_dir = tool_dir.join(v);
            fs::remove_dir_all(version_dir).ok();
            info!(name, version = %version, "tool version uninstalled");
        } else {
            // Remove all versions
            self.tools.remove(name);
            self.latest.remove(name);
            fs::remove_dir_all(&tool_dir).ok();
            info!(name, "tool uninstalled (all versions)");
        }

        Ok(())
    }

    /// Hot-reloads a tool by re-reading from disk and recompiling.
    ///
    /// Replaces the existing registration without interrupting in-flight calls.
    /// New calls will use the recompiled sandbox.
    #[instrument(skip(self), fields(name))]
    pub async fn reload(&self, name: &str) -> Result<()> {
        debug!("hot-reloading tool");

        // Find installed version
        let version = self
            .latest
            .get(name)
            .ok_or_else(|| {
                ToolError::NotFound(self.registry_path.join(name))
            })?
            .value()
            .clone();

        let version_dir = self.registry_path.join(name).join(version.to_string());
        let wasm_path = version_dir.join("tool.wasm");
        let manifest_path = version_dir.join("manifest.toml");

        if !wasm_path.exists() {
            return Err(ToolError::NotFound(wasm_path));
        }

        // Re-install (overwrites existing entry atomically)
        self.install(&wasm_path, Some(&manifest_path)).await?;

        info!(name, version = %version, "tool hot-reloaded");
        Ok(())
    }
    /// Scans the registry directory and loads all tools found.
    ///
    /// Expected layout: `{registry_path}/{name}/{version}/tool.wasm`
    /// Returns the number of tools successfully loaded.
    #[instrument(skip(self))]
    pub async fn load_from_directory(&self) -> Result<usize> {
        debug!("loading tools from registry directory");

        if !self.registry_path.exists() {
            return Ok(0);
        }

        let mut count = 0;
        let entries = fs::read_dir(&self.registry_path).map_err(ToolError::IoError)?;

        for entry in entries {
            let entry = entry.map_err(ToolError::IoError)?;
            let tool_dir = entry.path();
            if !tool_dir.is_dir() {
                continue;
            }

            let name = tool_dir.file_name().unwrap().to_string_lossy().to_string();

            let version_entries = fs::read_dir(&tool_dir).map_err(ToolError::IoError)?;
            for v_entry in version_entries {
                let v_entry = v_entry.map_err(ToolError::IoError)?;
                let version_dir = v_entry.path();
                if !version_dir.is_dir() {
                    continue;
                }

                let wasm_path = version_dir.join("tool.wasm");
                if wasm_path.exists() {
                    // Ignore errors for individual tools; continue loading others
                    if self
                        .install(&wasm_path, Some(&version_dir.join("manifest.toml")))
                        .await
                        .is_ok()
                    {
                        count += 1;
                    } else {
                        warn!(
                            name = %name,
                            version_dir = ?version_dir,
                            "failed to load tool version from directory"
                        );
                    }
                }            }
        }

        info!(count, "loaded tools from registry directory");
        Ok(count)
    }

    /// Returns the registry's root path.
    pub fn registry_path(&self) -> &Path {
        &self.registry_path
    }

    /// Returns the number of unique tool names registered.
    pub fn name_count(&self) -> usize {
        self.tools.len()
    }

    /// Returns the total number of installed tool versions.
    pub fn version_count(&self) -> usize {
        self.tools
            .iter()
            .map(|e| e.value().len())
            .sum()
    }
}

// =============================================================================
// Checksum Utility
// =============================================================================

/// Computes the SHA-256 hex digest of WASM binary bytes.
pub fn compute_wasm_checksum(bytes: &[u8]) -> String {
    use sha2::{Sha256, Digest};

    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let result = hasher.finalize();

    hex::encode(result)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    #[test]
    fn test_tool_version_ordering() {
        let v1 = ToolVersion { major: 1, minor: 0, patch: 0 };
        let v2 = ToolVersion { major: 1, minor: 0, patch: 1 };
        let v3 = ToolVersion { major: 1, minor: 1, patch: 0 };
        let v4 = ToolVersion { major: 2, minor: 0, patch: 0 };

        assert!(v1 < v2);
        assert!(v2 < v3);
        assert!(v3 < v4);
        assert_eq!(v1.to_string(), "1.0.0");
    }

    #[test]
    fn test_tool_version_parsing() {
        assert!(ToolVersion::parse("1.2.3").is_ok());
        assert!(ToolVersion::parse("v0.0.1").is_ok());
        assert!(ToolVersion::parse("invalid").is_err());
        assert!(ToolVersion::parse("1.2").is_err());
    }

    #[test]
    fn test_checksum_consistency() {
        let data = b"hello wasm";
        let h1 = compute_wasm_checksum(data);
        let h2 = compute_wasm_checksum(data);
        assert_eq!(h1, h2);
        assert_ne!(compute_wasm_checksum(b"other"), h1);
    }

    #[tokio::test]
    async fn test_registry_lifecycle() {
        let tmp = TempDir::new().unwrap();
        let registry = ToolRegistry::new(tmp.path().to_path_buf()).unwrap();

        // Initially empty
        assert_eq!(registry.name_count(), 0);
        assert!(registry.get("test", None).is_none());

        // Install would require a real WASM file; skip file ops for unit test
        // but verify DashMap operations
        let versions = registry.tools.entry("mock".into()).or_default();
        versions.insert(
            ToolVersion { major: 1, minor: 0, patch: 0 },
            Arc::new(ToolRegistration {
                manifest: ProtoManifest {
                    name: "mock".into(),
                    version: "1.0.0".into(),
                    description: "mock".into(),
                    author: None,                    license: None,
                    capabilities_required: vec![],
                    resource_limits: nexus_proto::tool::ResourceLimits::default(),
                    input_schema: serde_json::Value::Null,
                    output_schema: serde_json::Value::Null,
                    wasm_path: None,
                    checksum_sha256: None,
                },
                sandbox: Arc::new(WasmSandbox::new(
                    registry.engine.clone(),
                    ProtoManifest {
                        name: "mock".into(),
                        version: "1.0.0".into(),
                        description: "mock".into(),
                        author: None,
                        license: None,
                        capabilities_required: vec![],
                        resource_limits: nexus_proto::tool::ResourceLimits::default(),
                        input_schema: serde_json::Value::Null,
                        output_schema: serde_json::Value::Null,
                        wasm_path: None,
                        checksum_sha256: None,
                    },
                    vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00], // minimal wasm magic
                ).unwrap()),
                installed_at: Utc::now(),
                call_count: AtomicU64::new(5),
                error_count: AtomicU64::new(1),
                total_execution_ms: AtomicU64::new(1500),
            }),
        );
        registry.latest.insert("mock".into(), ToolVersion { major: 1, minor: 0, patch: 0 });

        // Verify get and stats
        let reg = registry.get("mock", None).unwrap();
        let stats = reg.stats();
        assert_eq!(stats.name, "mock");
        assert_eq!(stats.call_count, 5);
        assert_eq!(stats.error_count, 1);
        assert!((stats.avg_execution_ms - 300.0).abs() < 0.1);
        assert!((stats.success_rate - 0.8).abs() < 0.01);
    }
}
