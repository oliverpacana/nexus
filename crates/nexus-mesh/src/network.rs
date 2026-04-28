// crates/nexus-mesh/src/network.rs

use std::sync::Arc;

use libp2p::identity::Keypair;
use libp2p::multiaddr::Multiaddr;
use libp2p::swarm::behaviour::toggle::Toggle;
use libp2p::swarm::NetworkBehaviour;
use libp2p::{gossipsub, identify, mdns, request_response, swarm::SwarmEvent, Swarm, SwarmBuilder};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::blackboard::{Blackboard, BlackboardChange};
use crate::error::MeshError;
use crate::crdt::LWWMap;

// =============================================================================
// Mesh Protocol Messages
// =============================================================================

/// Requests sent between mesh nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MeshRequest {
    /// Request blackboard state delta since a logical timestamp.
    BlackboardSync { since_ts: u64 },
    /// Delegate a task/message to another node.
    DelegateTask { envelope: Vec<u8> },
    /// Ping to verify node liveness.
    HealthCheck,
    /// Query which nodes support a specific capability.
    CapabilityQuery { capability: String },
}

/// Responses returned from mesh nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MeshResponse {
    /// Serialized LWWMap delta for blackboard synchronization.
    BlackboardDelta { delta: Vec<u8> },
    /// Acknowledgement of a delegated task.
    TaskAck { envelope_id: String },
    /// Health check success response.
    HealthOk { node_id: String },
    /// List of node IDs that possess the queried capability.
    CapabilityNodes { node_ids: Vec<String> },
}

// =============================================================================
// Nexus Behaviour// =============================================================================

/// Composed libp2p NetworkBehaviour for the Nexus mesh.
/// Combines mDNS discovery, Gossipsub pub/sub, Request-Response sync, and Identify.
#[derive(NetworkBehaviour)]
pub struct NexusBehaviour {
    /// LAN peer discovery via multicast DNS.
    pub mdns: Toggle<mdns::tokio::Behaviour>,
    
    /// Publish/Subscribe for blackboard synchronization and announcements.
    pub gossipsub: gossipsub::Behaviour,
    
    /// Direct request-response protocol for state sync and delegation.
    pub request_response: request_response::Behaviour<
        request_response::codec::Cbor<MeshRequest, MeshResponse>
    >,
    
    /// Protocol and version exchange on connection establishment.
    pub identify: identify::Behaviour,
}

// =============================================================================
// Swarm Builder
// =============================================================================

/// Builds and configures the libp2p Swarm for a mesh node.
///
/// # Arguments
/// * `keypair` - Node identity keypair for TLS/Noise authentication.
/// * `listen_addr` - Multiaddr to listen on (e.g., `/ip4/0.0.0.0/tcp/9000`).
/// * `blackboard` - Shared blackboard instance for state synchronization.
/// * `mdns_enabled` - Whether to activate mDNS discovery.
///
/// # Returns
/// A fully configured `Swarm<NexusBehaviour>` ready to dial and listen.
pub async fn build_swarm(
    keypair: Keypair,
    listen_addr: Multiaddr,
    blackboard: Arc<Blackboard>,
    mdns_enabled: bool,
) -> Result<Swarm<NexusBehaviour>, MeshError> {
    // 1. Transport: TCP + Noise + Yamux
    let transport = libp2p::tcp::tokio::Transport::default()
        .upgrade(libp2p::core::upgrade::Version::V1)
        .authenticate(libp2p::noise::Config::new(&keypair)?)
        .multiplex(libp2p::yamux::Config::default())
        .boxed();

    // 2. Gossipsub configuration
    let gossipsub_config = gossipsub::ConfigBuilder::default()        .heartbeat_interval(std::time::Duration::from_secs(5))
        .validation_mode(gossipsub::ValidationMode::Permissive)
        .build()
        .map_err(|e| MeshError::Network(format!("Failed to build gossipsub config: {}", e)))?;

    let mut gossipsub = gossipsub::Behaviour::new(
        gossipsub::MessageAuthenticity::Signed(keypair.clone()),
        gossipsub_config,
    )
    .map_err(|e| MeshError::Network(format!("Failed to create gossipsub behaviour: {}", e)))?;

    // Subscribe to blackboard topic
    let bb_topic = gossipsub::Topic::new("nexus-blackboard");
    gossipsub.subscribe(&bb_topic).map_err(|e| MeshError::Network(format!("Failed to subscribe to blackboard topic: {}", e)))?;

    // 3. Request-Response configuration
    let req_res_config = request_response::Config::default();
    let request_response = request_response::Behaviour::new(
        request_response::codec::Cbor::default(),
        [(request_response::ProtocolSupport::Full, b"/nexus-sync/1".to_vec())],
        req_res_config,
    );

    // 4. Identify configuration
    let identify_config = identify::Config::new("nexus/0.1.0".into(), keypair.public())
        .with_interval(std::time::Duration::from_secs(60));
    let identify = identify::Behaviour::new(identify_config);

    // 5. mDNS (optional)
    let mdns_behaviour = if mdns_enabled {
        Toggle::On(
            mdns::tokio::Behaviour::new(
                mdns::Config::default(),
                keypair.public().to_peer_id(),
            )
            .map_err(|e| MeshError::Network(format!("Failed to create mDNS behaviour: {}", e)))?,
        )
    } else {
        Toggle::Off
    };

    // 6. Assemble Behaviour
    let behaviour = NexusBehaviour {
        mdns: mdns_behaviour,
        gossipsub,
        request_response,
        identify,
    };

    // 7. Build Swarm    let mut swarm = SwarmBuilder::with_tokio_executor()
        .with_transport(transport)
        .with_behaviour(|_| behaviour)
        .expect("Failed to build swarm behaviour")
        .build();

    // Listen on specified address
    swarm.listen_on(listen_addr.clone())
        .map_err(|e| MeshError::Network(format!("Failed to listen on {}: {}", listen_addr, e)))?;

    info!(addr = %listen_addr, "mesh swarm initialized and listening");
    Ok(swarm)
}
