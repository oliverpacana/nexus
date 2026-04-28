// crates/nexus-mesh/src/lib.rs

//! # Nexus Mesh
//!
//! Distributed P2P agent fabric for the Nexus AI agent runtime.
//!
//! `nexus-mesh` enables agents to discover each other, route work, and share state
//! across a partition-tolerant network without a central coordinator. It uses:
//! - **libp2p** for transport, security, and routing
//! - **CRDTs** for conflict-free shared blackboard synchronization
//! - **Gossipsub** for pub/sub state dissemination
//! - **mDNS** for zero-configuration LAN discovery
//!
//! ## Architecture
//!
//! Each node runs a `MeshNode` which manages:
//! - A libp2p `Swarm` handling connections and protocols
//! - A `Blackboard` CRDT-backed key-value store for shared state
//! - Capability announcement and query mechanisms
//! - Periodic delta synchronization with peers
//!
//! ## Usage
//!
//! ```rust
//! use nexus_mesh::{MeshNode, MeshNodeConfig, BlackboardEntry};
//! use nexus_proto::agent::AgentId;
//!
//! #[tokio::main]
//! async fn main() {
//!     let config = MeshNodeConfig {
//!         node_id: "node-alpha".into(),
//!         listen_addr: "/ip4/0.0.0.0/tcp/9000".into(),
//!         mdns_enabled: true,
//!         blackboard_sync_interval_ms: 5000,
//!     };
//!
//!     let mut node = MeshNode::new(config).await.unwrap();
//!     node.start().await.unwrap();
//!
//!     // Post to the shared blackboard
//!     let entry = BlackboardEntry {
//!         value: serde_json::json!({"status": "online"}),
//!         author_id: AgentId::new(),
//!         node_id: "node-alpha".into(),
//!         scope: nexus_mesh::blackboard::BlackboardScope::Cluster,
//!         expires_at: None,
//!         tags: vec!["heartbeat".into()],
//!     };
//!     node.blackboard().set("status::node-alpha".into(), entry).await;
//! }//! ```

#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]
#![allow(clippy::type_complexity)]

// =============================================================================
// Module Declarations
// =============================================================================

pub mod crdt;
pub mod blackboard;
pub mod network;
pub mod discovery;
pub mod node;

// =============================================================================
// Error Type
// =============================================================================

/// Errors that can occur in the mesh subsystem.
#[derive(Debug, thiserror::Error)]
pub enum MeshError {
    #[error("network error: {0}")]
    Network(String),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("libp2p error: {0}")]
    Libp2p(#[from] libp2p::swarm::dial_opts::DialError),
}

// =============================================================================
// Public Re-Exports
// =============================================================================

pub use node::{MeshNode, MeshNodeConfig};
pub use blackboard::{Blackboard, BlackboardEntry, BlackboardScope, BlackboardChange};
pub use network::{NexusBehaviour, MeshRequest, MeshResponse};
pub use discovery::DiscoveryConfig;
pub use error::MeshError;

// Convenience prelude
pub mod prelude {    pub use crate::node::{MeshNode, MeshNodeConfig};
    pub use crate::blackboard::{Blackboard, BlackboardEntry, BlackboardScope};
    pub use crate::crdt::{LamportClock, LWWMap, LWWValue};
    pub use crate::error::MeshError;
}
