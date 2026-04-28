// crates/nexus-mesh/src/discovery.rs

use libp2p::Swarm;
use tracing::{debug, info, warn};

use crate::network::NexusBehaviour;
use crate::error::MeshError;

/// Configuration for mesh discovery mechanisms.
#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    /// Whether to enable mDNS for LAN auto-discovery.
    pub mdns_enabled: bool,
    
    /// Optional rendezvous server multiaddr for WAN discovery.
    pub rendezvous_addr: Option<String>,
}

/// Starts or processes mDNS discovery for the mesh.
/// 
/// In the context of the `MeshNode` event loop, this function is called 
/// when an mDNS event is emitted. It handles peer dialing and logging.
/// If used as a standalone initializer, it verifies mDNS is active.
pub async fn start_mdns_discovery(
    swarm: &mut Swarm<NexusBehaviour>,
) -> Result<(), MeshError> {
    // Verify mDNS is enabled in the behaviour
    match &swarm.behaviour().mdns {
        libp2p::swarm::behaviour::toggle::Toggle::On(_) => {
            debug!("mDNS discovery is active");
            Ok(())
        }
        libp2p::swarm::behaviour::toggle::Toggle::Off => {
            warn!("mDNS discovery is disabled in mesh configuration");
            Ok(())
        }
    }
}

/// Handles a discovered peer list from mDNS and initiates dials.
/// Called internally by the node's swarm event loop.
pub fn handle_mdns_discovered(
    swarm: &mut Swarm<NexusBehaviour>,
    peers: Vec<(libp2p::PeerId, libp2p::Multiaddr)>,
) {
    let mut dialed = 0;
    for (peer_id, addr) in peers {
        // Construct dial address with peer ID
        if let Some(dial_addr) = addr.with_p2p(peer_id).ok() {
            if swarm.dial(dial_addr).is_ok() {
                dialed += 1;
                debug!(%peer_id, "dialed newly discovered mDNS peer");
            }
        }
    }
    if dialed > 0 {
        info!(count = dialed, "initiated connections to mDNS peers");
    }
}
