// crates/nexus-mesh/src/node.rs

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use libp2p::gossipsub::{Event as GossipsubEvent, MessageId, Topic};
use libp2p::identify::Event as IdentifyEvent;
use libp2p::mdns::Event as MdnsEvent;
use libp2p::request_response::{Event as RequestResponseEvent, Message, ResponseChannel};
use libp2p::swarm::{Swarm, SwarmEvent};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, RwLock};
use tokio::task::JoinHandle;
use tokio::time::{sleep, interval};
use tracing::{debug, error, info, warn, instrument};
use uuid::Uuid;

use crate::blackboard::{Blackboard, BlackboardChange};
use crate::crdt::LWWMap;
use crate::discovery::handle_mdns_discovered;
use crate::error::MeshError;
use crate::network::{build_swarm, MeshRequest, MeshResponse, NexusBehaviour};

// =============================================================================
// Mesh Node Configuration
// =============================================================================

/// Configuration for a running mesh node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshNodeConfig {
    /// Unique identifier for this node in the mesh.
    pub node_id: String,
    
    /// Address to bind the libp2p swarm (e.g., "/ip4/0.0.0.0/tcp/9000").
    pub listen_addr: String,
    
    /// Whether to enable mDNS for LAN peer discovery.
    pub mdns_enabled: bool,
    
    /// Interval in milliseconds for periodic blackboard synchronization broadcasts.
    pub blackboard_sync_interval_ms: u64,
}

// =============================================================================
// Mesh Node
// =============================================================================

/// The running mesh node managing P2P networking, discovery, and blackboard sync.
pub struct MeshNode {    pub config: MeshNodeConfig,
    pub blackboard: Arc<Blackboard>,
    pub peer_count: Arc<AtomicUsize>,
    pub running: Arc<AtomicBool>,
    pub task_handle: Option<JoinHandle<()>>,
    pub swarm: Mutex<Option<Swarm<NexusBehaviour>>>,
}

impl MeshNode {
    /// Creates a new mesh node instance.
    /// Does not start networking until `start()` is called.
    pub async fn new(config: MeshNodeConfig) -> Result<Self, MeshError> {
        let (blackboard, _) = Blackboard::new(config.node_id.clone());
        
        Ok(Self {
            config,
            blackboard: Arc::new(blackboard),
            peer_count: Arc::new(AtomicUsize::new(0)),
            running: Arc::new(AtomicBool::new(false)),
            task_handle: None,
            swarm: Mutex::new(None),
        })
    }

    /// Starts the mesh node, spawning the background event loop.
    pub async fn start(&mut self) -> Result<(), MeshError> {
        if self.running.load(Ordering::SeqCst) {
            return Ok(());
        }

        let keypair = libp2p::identity::Keypair::generate_ed25519();
        let listen_addr = self.config.listen_addr.parse()
            .map_err(|e| MeshError::Network(format!("Invalid listen address: {}", e)))?;

        let swarm = build_swarm(
            keypair,
            listen_addr,
            Arc::clone(&self.blackboard),
            self.config.mdns_enabled,
        ).await?;

        *self.swarm.lock().await = Some(swarm);
        self.running.store(true, Ordering::SeqCst);

        // Clone fields for background task
        let swarm = self.swarm.lock().await.take().unwrap();
        let bb = Arc::clone(&self.blackboard);
        let peer_count = Arc::clone(&self.peer_count);
        let running = Arc::clone(&self.running);
        let sync_interval_ms = self.config.blackboard_sync_interval_ms;        let node_id = self.config.node_id.clone();

        // Spawn background event loop
        let handle = tokio::spawn(async move {
            Self::run_event_loop(
                swarm,
                bb,
                peer_count,
                running,
                sync_interval_ms,
                node_id,
            ).await;
        });

        self.task_handle = Some(handle);
        info!(node_id = %self.config.node_id, "mesh node started");
        Ok(())
    }

    /// Core swarm event loop handling networking, sync, and discovery.
    async fn run_event_loop(
        mut swarm: Swarm<NexusBehaviour>,
        blackboard: Arc<Blackboard>,
        peer_count: Arc<AtomicUsize>,
        running: Arc<AtomicBool>,
        sync_interval_ms: u64,
        node_id: String,
    ) {
        let mut sync_timer = interval(Duration::from_millis(sync_interval_ms));
        let bb_topic = Topic::new("nexus-blackboard");

        loop {
            if !running.load(Ordering::SeqCst) {
                info!("mesh node shutting down");
                break;
            }

            tokio::select! {
                // 1. Poll Swarm Events
                event = swarm.select_next_some() => {
                    match event {
                        SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                            peer_count.fetch_add(1, Ordering::SeqCst);
                            debug!(%peer_id, "connection established");
                        }
                        SwarmEvent::ConnectionClosed { peer_id, .. } => {
                            peer_count.fetch_sub(1, Ordering::SeqCst);
                            debug!(%peer_id, "connection closed");
                        }
                        SwarmEvent::Behaviour(NexusBehaviourEvent::Mdns(mdns::Event::Discovered(list))) => {                            handle_mdns_discovered(&mut swarm, list);
                        }
                        SwarmEvent::Behaviour(NexusBehaviourEvent::Gossipsub(GossipsubEvent::Message { message, .. })) => {
                            // Parse blackboard delta and merge
                            if let Ok(delta_json) = serde_json::from_slice::<Vec<u8>>(&message.payload) {
                                // The payload is actually a serialized LWWMap. 
                                // In practice, we'd serialize/deserialize the LWWMap properly.
                                // Here we assume the payload contains the delta bytes.
                                // For simplicity in this prototype, we parse directly if possible.
                                // A real impl would use bincode/cbor for the payload.
                            }
                            
                            // Handle blackboard sync messages
                            if message.topic == bb_topic.hash() {
                                if let Ok(delta_bytes) = serde_json::from_slice::<Vec<u8>>(&message.payload) {
                                    // Deserialize LWWMap (assuming serde compatibility for demo)
                                    // In production, use a robust codec like bincode
                                    if let Ok(delta) = serde_json::from_slice::<LWWMap>(&delta_bytes) {
                                        let remote_ts = 0; // Would extract from message metadata
                                        blackboard.merge_delta(delta, remote_ts).await;
                                    }
                                }
                            }
                        }
                        SwarmEvent::Behaviour(NexusBehaviourEvent::RequestResponse(RequestResponseEvent::Message { peer, message, .. })) => {
                            match message {
                                Message::Request { request, channel, .. } => {
                                    Self::handle_request(&mut swarm, peer, request, channel, &blackboard).await;
                                }
                                Message::Response { .. } => {
                                    // Handle responses if needed
                                }
                            }
                        }
                        SwarmEvent::Behaviour(NexusBehaviourEvent::Identify(IdentifyEvent::Received { peer_id, info, .. })) => {
                            // Record peer capabilities/versions
                            debug!(%peer_id, protocol = %info.protocol_version, "peer identified");
                        }
                        _ => {}
                    }
                }
                
                // 2. Periodic Blackboard Sync Broadcast
                _ = sync_timer.tick() => {
                    Self::broadcast_blackboard_delta(&mut swarm, &blackboard, &bb_topic).await;
                }
            }
        }
    }
    /// Handles incoming request-response messages.
    async fn handle_request(
        swarm: &mut Swarm<NexusBehaviour>,
        peer: libp2p::PeerId,
        request: MeshRequest,
        channel: ResponseChannel<MeshResponse>,
        blackboard: &Blackboard,
    ) {
        let response = match request {
            MeshRequest::BlackboardSync { since_ts } => {
                let delta = blackboard.delta_since(since_ts).await;
                let delta_bytes = serde_json::to_vec(&delta).unwrap_or_default();
                MeshResponse::BlackboardDelta { delta: delta_bytes }
            }
            MeshRequest::HealthCheck => {
                MeshResponse::HealthOk { node_id: "current-node".into() }
            }
            MeshRequest::CapabilityQuery { capability } => {
                let nodes = blackboard.nodes_with_capability(&capability).await;
                MeshResponse::CapabilityNodes { node_ids: nodes }
            }
            MeshRequest::DelegateTask { envelope } => {
                // Logic to route task to local agent kernel would go here
                MeshResponse::TaskAck { envelope_id: "acked".into() }
            }
        };

        if let Err(e) = swarm.behaviour_mut().request_response.send_response(channel, response) {
            warn!(%peer, error = %e, "failed to send response to peer");
        }
    }

    /// Broadcasts the latest blackboard delta to the mesh via Gossipsub.
    async fn broadcast_blackboard_delta(
        swarm: &mut Swarm<NexusBehaviour>,
        blackboard: &Blackboard,
        topic: &Topic,
    ) {
        // In a real implementation, track last sync TS to avoid broadcasting full state
        // Here we broadcast a heartbeat/empty delta for demonstration
        let payload = vec![]; // Simplified
        if let Err(e) = swarm.behaviour_mut().gossipsub.publish(topic.hash(), payload) {
            debug!(error = %e, "failed to publish blackboard delta");
        }
    }

    /// Returns a reference to the node's blackboard.
    pub fn blackboard(&self) -> Arc<Blackboard> {
        Arc::clone(&self.blackboard)
    }
    /// Returns the current number of connected peers.
    pub fn peer_count(&self) -> usize {
        self.peer_count.load(Ordering::SeqCst)
    }

    /// Gracefully shuts down the mesh node.
    pub async fn shutdown(&self) {
        self.running.store(false, Ordering::SeqCst);
        if let Some(handle) = &self.task_handle {
            handle.abort();
        }
        info!("mesh node shutdown signal sent");
    }

    /// Returns whether the node's background task is running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}
