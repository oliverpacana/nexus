use std::collections::HashSet;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use dashmap::DashMap;
use nexus_proto::agent::AgentId;
use nexus_proto::memory::{MemoryAccess, MemoryScope, MemoryTier};
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument};

use crate::embeddings::MemoryError;

pub type Result<T> = std::result::Result<T, MemoryError>;

// =============================================================================
// MemoryPermission — A Grant of Access
// =============================================================================

/// Represents an explicit permission grant from one agent to another.
/// Grants are scoped to a specific memory tier, access level, and memory scope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPermission {
    /// The agent that granted this permission.
    pub grantor: AgentId,

    /// The agent that receives this permission.
    pub grantee: AgentId,

    /// Which memory tier this grant applies to.
    pub tier: MemoryTier,

    /// What kind of access is granted (Read, Write, or ReadWrite).
    pub access: MemoryAccess,

    /// Which memory scope this grant covers.
    pub scope: MemoryScope,

    /// Optional expiration timestamp; grant is invalid after this time.
    pub expires_at: Option<DateTime<Utc>>,
}

impl MemoryPermission {
    /// Creates a new permission grant.
    pub fn new(
        grantor: AgentId,
        grantee: AgentId,
        tier: MemoryTier,
        access: MemoryAccess,
        scope: MemoryScope,
        expires_at: Option<DateTime<Utc>>,    ) -> Self {
        Self {
            grantor,
            grantee,
            tier,
            access,
            scope,
            expires_at,
        }
    }

    /// Returns `true` if this grant has expired.
    pub fn is_expired(&self) -> bool {
        self.expires_at
            .map(|exp| Utc::now() > exp)
            .unwrap_or(false)
    }

    /// Returns `true` if this grant permits the specified access type.
    pub fn permits(&self, requested: MemoryAccess) -> bool {
        match (self.access, requested) {
            // ReadWrite permits both
            (MemoryAccess::ReadWrite, _) => true,
            // Read permits only read
            (MemoryAccess::Read, MemoryAccess::Read) => true,
            // Write permits only write
            (MemoryAccess::Write, MemoryAccess::Write) => true,
            // Mismatched or insufficient
            _ => false,
        }
    }

    /// Returns `true` if this grant applies to the given tier and scope.
    pub fn matches(&self, tier: MemoryTier, scope: MemoryScope) -> bool {
        self.tier == tier && self.scope == scope
    }
}

// =============================================================================
// GrantTable — Active Permission Registry
// =============================================================================

/// Stores all active memory permission grants.
/// Indexed by grantee for efficient lookup during access checks.
///
/// # Thread Safety
/// - Uses `DashMap` for lock-free concurrent reads
/// - Writes acquire per-key locks, minimizing contention
/// - All operations are atomic at the grant level
pub struct GrantTable {    /// Map: grantee → list of grants they hold
    grants: DashMap<AgentId, Vec<MemoryPermission>>,

    /// Reverse index: grantor → set of grantees (for efficient revocation)
    by_grantor: DashMap<AgentId, HashSet<AgentId>>,
}

impl Default for GrantTable {
    fn default() -> Self {
        Self::new()
    }
}

impl GrantTable {
    /// Creates a new empty grant table.
    pub fn new() -> Self {
        Self {
            grants: DashMap::new(),
            by_grantor: DashMap::new(),
        }
    }

    /// Grants a new permission to an agent.
    ///
    /// # Notes
    /// - Duplicate grants are allowed (idempotent)
    /// - Expired grants are not automatically filtered; call `prune_expired()` periodically
    #[instrument(skip(self), fields(grantor = %perm.grantor, grantee = %perm.grantee, tier = ?perm.tier))]
    pub fn grant(&self, perm: MemoryPermission) {
        let grantee = perm.grantee;
        let grantor = perm.grantor;

        // Add to grantee's list
        self.grants
            .entry(grantee)
            .or_default()
            .push(perm.clone());

        // Update reverse index
        self.by_grantor
            .entry(grantor)
            .or_default()
            .insert(grantee);

        debug!(
            grantor = %grantor,
            grantee = %grantee,
            tier = ?perm.tier,
            access = ?perm.access,
            scope = ?perm.scope,            "permission granted"
        );
    }

    /// Revokes a specific grant from a grantor to a grantee for a tier.
    ///
    /// # Arguments
    /// * `grantee` - The agent whose permission is being revoked
    /// * `grantor` - The agent who originally granted the permission
    /// * `tier` - The memory tier to revoke access for
    ///
    /// # Behavior
    /// Removes all grants matching (grantee, grantor, tier). Other grants are unaffected.
    #[instrument(skip(self), fields(grantee = %grantee, grantor = %grantor, tier = ?tier))]
    pub fn revoke(&self, grantee: AgentId, grantor: AgentId, tier: MemoryTier) {
        if let Some(mut grants) = self.grants.get_mut(&grantee) {
            let before = grants.len();
            grants.retain(|g| !(g.grantor == grantor && g.tier == tier));
            let removed = before - grants.len();

            if removed > 0 {
                debug!(
                    grantee = %grantee,
                    grantor = %grantor,
                    tier = ?tier,
                    removed,
                    "permissions revoked"
                );
            }
        }

        // Clean up reverse index
        if let Some(mut grantees) = self.by_grantor.get_mut(&grantor) {
            grantees.remove(&grantee);
            if grantees.is_empty() {
                drop(grantees);
                self.by_grantor.remove(&grantor);
            }
        }
    }

    /// Revokes all permissions involving a specific agent.
    ///
    /// # Arguments
    /// * `agent_id` - The agent to revoke permissions for
    ///
    /// # Behavior
    /// - Removes all grants WHERE this agent is the grantee
    /// - Removes all grants WHERE this agent is the grantor
    /// - Cleans up reverse index entries    #[instrument(skip(self), fields(agent_id = %agent_id))]
    pub fn revoke_all_for_agent(&self, agent_id: AgentId) {
        // Remove all grants where agent is grantee
        if self.grants.remove(&agent_id).is_some() {
            debug!(agent_id = %agent_id, "revoked all grants received by agent");
        }

        // Remove all grants where agent is grantor
        if let Some(grantees) = self.by_grantor.remove(&agent_id) {
            let (_, granted_to) = grantees;
            for grantee in granted_to {
                if let Some(mut grants) = self.grants.get_mut(&grantee) {
                    grants.retain(|g| g.grantor != agent_id);
                }
            }
            debug!(
                agent_id = %agent_id,
                grant_count = granted_to.len(),
                "revoked all grants issued by agent"
            );
        }
    }

    /// Checks if an agent has READ access to memory owned by another agent.
    ///
    /// # Arguments
    /// * `agent` - The agent requesting access
    /// * `owner` - The agent that owns the memory entry
    /// * `tier` - The memory tier being accessed
    /// * `scope` - The scope of the memory entry
    ///
    /// # Returns
    /// `true` if access is permitted, `false` otherwise.
    #[instrument(skip(self), fields(agent = %agent, owner = %owner, tier = ?tier, scope = ?scope))]
    pub fn check_read(
        &self,
        agent: AgentId,
        owner: AgentId,
        tier: MemoryTier,
        scope: MemoryScope,
    ) -> bool {
        // Owner always has implicit read access to their own memory
        if agent == owner {
            return true;
        }

        // Global scope: anyone can read
        if scope == MemoryScope::Global {
            return true;
        }
        // Private scope: only owner can read (already handled above)
        if scope == MemoryScope::Private {
            return false;
        }

        // Group scope: check explicit grants
        // (Group membership is checked by MemoryAccessChecker, not here)
        if let Some(grants) = self.grants.get(&agent) {
            for grant in grants.iter() {
                if !grant.is_expired()
                    && grant.grantor == owner
                    && grant.matches(tier, scope)
                    && grant.permits(MemoryAccess::Read)
                {
                    return true;
                }
            }
        }

        false
    }

    /// Checks if an agent has WRITE access to memory owned by another agent.
    ///
    /// # Arguments
    /// Same as `check_read`.
    ///
    /// # Returns
    /// `true` if write access is permitted, `false` otherwise.
    #[instrument(skip(self), fields(agent = %agent, owner = %owner, tier = ?tier, scope = ?scope))]
    pub fn check_write(
        &self,
        agent: AgentId,
        owner: AgentId,
        tier: MemoryTier,
        scope: MemoryScope,
    ) -> bool {
        // Owner always has implicit write access to their own memory
        if agent == owner {
            return true;
        }

        // Private scope: only owner can write
        if scope == MemoryScope::Private {
            return false;
        }

        // For Group and Global scopes, require explicit write grant
        if let Some(grants) = self.grants.get(&agent) {            for grant in grants.iter() {
                if !grant.is_expired()
                    && grant.grantor == owner
                    && grant.matches(tier, scope)
                    && grant.permits(MemoryAccess::Write)
                {
                    return true;
                }
            }
        }

        false
    }

    /// Removes all expired grants from the table.
    ///
    /// Call this periodically to prevent memory growth from expired grants.
    #[instrument(skip(self))]
    pub fn prune_expired(&self) {
        let mut pruned = 0;

        // Iterate over all grantees
        for mut entry in self.grants.iter_mut() {
            let before = entry.value().len();
            entry.value_mut().retain(|g| !g.is_expired());
            let removed = before - entry.value().len();
            pruned += removed;

            // Clean up empty lists
            if entry.value().is_empty() {
                let grantee = *entry.key();
                drop(entry);
                self.grants.remove(&grantee);
            }
        }

        // Clean up reverse index
        for mut entry in self.by_grantor.iter_mut() {
            entry.value_mut().retain(|grantee| {
                self.grants
                    .get(grantee)
                    .map_or(false, |grants| grants.iter().any(|g| g.grantor == *entry.key()))
            });
            if entry.value().is_empty() {
                let grantor = *entry.key();
                drop(entry);
                self.by_grantor.remove(&grantor);
            }
        }
        if pruned > 0 {
            debug!(pruned, "expired grants pruned");
        }
    }

    /// Lists all active (non-expired) grants held by a specific grantee.
    #[instrument(skip(self), fields(grantee = %grantee))]
    pub fn list_grants_for(&self, grantee: AgentId) -> Vec<MemoryPermission> {
        self.grants
            .get(&grantee)
            .map(|grants| {
                grants
                    .iter()
                    .filter(|g| !g.is_expired())
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Returns the total number of active grants in the table.
    pub fn grant_count(&self) -> usize {
        self.grants
            .iter()
            .map(|entry| entry.value().iter().filter(|g| !g.is_expired()).count())
            .sum()
    }

    /// Returns all grantees that have received grants from a specific grantor.
    pub fn grantees_of(&self, grantor: AgentId) -> Vec<AgentId> {
        self.by_grantor
            .get(&grantor)
            .map(|set| set.iter().copied().collect())
            .unwrap_or_default()
    }
}

// =============================================================================
// GroupTable — Supervisor Group Membership
// =============================================================================

/// Tracks which agents belong to which supervisor groups.
/// Used for Group-scope memory access decisions.
///
/// # Design
/// - An agent can belong to multiple groups
/// - Group membership is set by the kernel/supervisor, not by agents themselves
/// - Lookups are O(1) average case via DashMap
pub struct GroupTable {
    /// Map: agent_id → set of group names they belong to    memberships: DashMap<AgentId, HashSet<String>>,

    /// Reverse index: group_name → set of member agent IDs
    by_group: DashMap<String, HashSet<AgentId>>,
}

impl Default for GroupTable {
    fn default() -> Self {
        Self::new()
    }
}

impl GroupTable {
    /// Creates a new empty group table.
    pub fn new() -> Self {
        Self {
            memberships: DashMap::new(),
            by_group: DashMap::new(),
        }
    }

    /// Adds an agent to a supervisor group.
    #[instrument(skip(self), fields(agent = %agent, group = %group))]
    pub fn add_to_group(&self, agent: AgentId, group: String) {
        self.memberships
            .entry(agent)
            .or_default()
            .insert(group.clone());

        self.by_group
            .entry(group.clone())
            .or_default()
            .insert(agent);

        debug!(agent = %agent, group = %group, "agent added to group");
    }

    /// Removes an agent from a supervisor group.
    #[instrument(skip(self), fields(agent = %agent, group = %group))]
    pub fn remove_from_group(&self, agent: AgentId, group: String) {
        if let Some(mut groups) = self.memberships.get_mut(&agent) {
            groups.remove(&group);
            if groups.is_empty() {
                drop(groups);
                self.memberships.remove(&agent);
            }
        }

        if let Some(mut members) = self.by_group.get_mut(&group) {
            members.remove(&agent);            if members.is_empty() {
                drop(members);
                self.by_group.remove(&group);
            }
        }

        debug!(agent = %agent, group = %group, "agent removed from group");
    }

    /// Returns `true` if two agents share at least one supervisor group.
    #[instrument(skip(self), fields(a = %a, b = %b))]
    pub fn in_same_group(&self, a: AgentId, b: AgentId) -> bool {
        // Fast path: same agent is trivially in same group
        if a == b {
            return true;
        }

        // Get groups for both agents
        let groups_a = self.memberships.get(&a);
        let groups_b = self.memberships.get(&b);

        match (groups_a, groups_b) {
            (Some(a_groups), Some(b_groups)) => {
                a_groups.iter().any(|g| b_groups.contains(g))
            }
            _ => false,
        }
    }

    /// Returns all groups that an agent belongs to.
    #[instrument(skip(self), fields(agent = %agent))]
    pub fn agent_groups(&self, agent: AgentId) -> Vec<String> {
        self.memberships
            .get(&agent)
            .map(|groups| groups.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Returns all members of a specific group.
    #[instrument(skip(self), fields(group = %group_name))]
    pub fn group_members(&self, group_name: &str) -> Vec<AgentId> {
        self.by_group
            .get(group_name)
            .map(|members| members.iter().copied().collect())
            .unwrap_or_default()
    }

    /// Removes an agent from all groups.
    #[instrument(skip(self), fields(agent = %agent_id))]
    pub fn remove_agent_from_all_groups(&self, agent_id: AgentId) {        if let Some(groups) = self.memberships.remove(&agent_id) {
            let (_, agent_groups) = groups;
            for group in agent_groups {
                if let Some(mut members) = self.by_group.get_mut(&group) {
                    members.remove(&agent_id);
                    if members.is_empty() {
                        drop(members);
                        self.by_group.remove(&group);
                    }
                }
            }
            debug!(agent_id = %agent_id, group_count = agent_groups.len(), "agent removed from all groups");
        }
    }

    /// Returns the total number of unique groups.
    pub fn group_count(&self) -> usize {
        self.by_group.len()
    }

    /// Returns the total number of agents with group memberships.
    pub fn agent_count(&self) -> usize {
        self.memberships.len()
    }
}

// =============================================================================
// MemoryAccessChecker — Enforcement Point
// =============================================================================

/// The central enforcement point for memory access decisions.
/// Used by all memory tier implementations (L1-L4) to authorize access.
///
/// # Security Model
/// - Private scope: only the owner can access (read or write)
/// - Group scope: agents in the same supervisor group can access, OR agents with explicit grants
/// - Global scope: anyone can read; write requires explicit grant from owner
/// - L1 (Working) and L2 (Episodic) are always treated as Private for writes
/// - L3 (Semantic) and L4 (Procedural) respect scope rules for both reads and writes
pub struct MemoryAccessChecker {
    /// The grant table containing explicit permissions.
    grant_table: Arc<GrantTable>,

    /// The group membership table for Group-scope decisions.
    group_table: Arc<GroupTable>,
}

impl MemoryAccessChecker {
    /// Creates a new access checker with the given grant and group tables.
    pub fn new(grant_table: Arc<GrantTable>, group_table: Arc<GroupTable>) -> Self {        Self {
            grant_table,
            group_table,
        }
    }

    /// Returns a reference to the underlying grant table.
    pub fn grant_table(&self) -> &Arc<GrantTable> {
        &self.grant_table
    }

    /// Returns a reference to the underlying group table.
    pub fn group_table(&self) -> &Arc<GroupTable> {
        &self.group_table
    }

    /// Checks if an agent can READ memory owned by another agent.
    ///
    /// # Decision Logic
    /// 1. If requestor == owner: ALLOW (implicit ownership)
    /// 2. If scope == Global: ALLOW (global reads are public)
    /// 3. If scope == Private: DENY (only owner can read private memory)
    /// 4. If scope == Group:
    ///    a. If requestor and owner share a supervisor group: ALLOW
    ///    b. Else if explicit read grant exists: ALLOW
    ///    c. Else: DENY
    ///
    /// # Returns
    /// - `Ok(())` if access is permitted
    /// - `Err(MemoryError::AccessDenied)` if access is denied
    #[instrument(skip(self), fields(requestor = %requestor, target_owner = %target_owner, tier = ?tier, scope = ?scope))]
    pub fn check_read(
        &self,
        requestor: AgentId,
        target_owner: AgentId,
        scope: MemoryScope,
        tier: MemoryTier,
    ) -> Result<()> {
        // Rule 1: Owner always has implicit read access
        if requestor == target_owner {
            return Ok(());
        }

        match scope {
            // Rule 2: Global scope allows anyone to read
            MemoryScope::Global => Ok(()),

            // Rule 3: Private scope denies all non-owners
            MemoryScope::Private => Err(MemoryError::AccessDenied {
                agent_id: requestor,                owner_id: target_owner,
                tier,
                scope,
                operation: "read".to_string(),
            }),

            // Rule 4: Group scope checks membership or explicit grant
            MemoryScope::Group => {
                // Check group membership first (fast path)
                if self.group_table.in_same_group(requestor, target_owner) {
                    debug!(
                        requestor = %requestor,
                        target_owner = %target_owner,
                        "group membership grants read access"
                    );
                    return Ok(());
                }

                // Check explicit grants
                if self.grant_table.check_read(requestor, target_owner, tier, scope) {
                    debug!(
                        requestor = %requestor,
                        target_owner = %target_owner,
                        "explicit grant grants read access"
                    );
                    return Ok(());
                }

                Err(MemoryError::AccessDenied {
                    agent_id: requestor,
                    owner_id: target_owner,
                    tier,
                    scope,
                    operation: "read".to_string(),
                })
            }
        }
    }

    /// Checks if an agent can WRITE to memory owned by another agent.
    ///
    /// # Decision Logic
    /// 1. If requestor == owner: ALLOW (implicit ownership)
    /// 2. If tier == L1 or L2: DENY for non-owners (working/episodic always private for writes)
    /// 3. If scope == Private: DENY (only owner can write private memory)
    /// 4. If scope == Group or Global:
    ///    a. If explicit write grant exists from owner to requestor: ALLOW
    ///    b. Else: DENY (group membership alone doesn't grant write access)
    ///
    /// # Returns    /// - `Ok(())` if access is permitted
    /// - `Err(MemoryError::AccessDenied)` if access is denied
    #[instrument(skip(self), fields(requestor = %requestor, target_owner = %target_owner, tier = ?tier, scope = ?scope))]
    pub fn check_write(
        &self,
        requestor: AgentId,
        target_owner: AgentId,
        scope: MemoryScope,
        tier: MemoryTier,
    ) -> Result<()> {
        // Rule 1: Owner always has implicit write access
        if requestor == target_owner {
            return Ok(());
        }

        // Rule 2: L1 and L2 are always private for writes (no sharing)
        if matches!(tier, MemoryTier::Working | MemoryTier::Episodic) {
            return Err(MemoryError::AccessDenied {
                agent_id: requestor,
                owner_id: target_owner,
                tier,
                scope,
                operation: "write".to_string(),
            });
        }

        match scope {
            // Rule 3: Private scope denies all non-owners
            MemoryScope::Private => Err(MemoryError::AccessDenied {
                agent_id: requestor,
                owner_id: target_owner,
                tier,
                scope,
                operation: "write".to_string(),
            }),

            // Rule 4: Group and Global scopes require explicit write grant
            MemoryScope::Group | MemoryScope::Global => {
                if self.grant_table.check_write(requestor, target_owner, tier, scope) {
                    debug!(
                        requestor = %requestor,
                        target_owner = %target_owner,
                        tier = ?tier,
                        scope = ?scope,
                        "explicit grant grants write access"
                    );
                    return Ok(());
                }

                Err(MemoryError::AccessDenied {                    agent_id: requestor,
                    owner_id: target_owner,
                    tier,
                    scope,
                    operation: "write".to_string(),
                })
            }
        }
    }

    /// Checks if an agent can perform the specified access operation.
    ///
    /// Convenience method that dispatches to `check_read` or `check_write`.
    pub fn check(
        &self,
        requestor: AgentId,
        target_owner: AgentId,
        scope: MemoryScope,
        tier: MemoryTier,
        access: MemoryAccess,
    ) -> Result<()> {
        match access {
            MemoryAccess::Read => self.check_read(requestor, target_owner, scope, tier),
            MemoryAccess::Write => self.check_write(requestor, target_owner, scope, tier),
            MemoryAccess::ReadWrite => {
                self.check_read(requestor, target_owner, scope, tier)?;
                self.check_write(requestor, target_owner, scope, tier)
            }
        }
    }

    /// Prunes expired grants from the grant table.
    /// Should be called periodically by a background task.
    pub fn prune_expired_grants(&self) {
        self.grant_table.prune_expired();
    }

    /// Returns a summary of permission state for observability.
    pub fn summary(&self) -> PermissionSummary {
        PermissionSummary {
            total_grants: self.grant_table.grant_count(),
            total_groups: self.group_table.group_count(),
            agents_with_groups: self.group_table.agent_count(),
        }
    }
}

/// Summary statistics for permission state.
#[derive(Debug, Clone)]
pub struct PermissionSummary {    pub total_grants: usize,
    pub total_groups: usize,
    pub agents_with_groups: usize,
}

// =============================================================================
// MemoryError Extension for Access Control
// =============================================================================

// Note: MemoryError is defined in crate::embeddings, but we need an access-specific variant.
// We extend it here via a new error type that can be converted to the main MemoryError.

/// Access control-specific error details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessDeniedDetails {
    pub agent_id: AgentId,
    pub owner_id: AgentId,
    pub tier: MemoryTier,
    pub scope: MemoryScope,
    pub operation: String,
}

impl std::fmt::Display for AccessDeniedDetails {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "access denied: agent '{}' cannot {} {} memory (tier={:?}, scope={:?}) owned by '{}'",
            self.agent_id, self.operation, self.tier, self.scope, self.owner_id
        )
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_agent_id() -> AgentId {
        AgentId::new()
    }

    #[test]
    fn test_permission_is_expired() {
        let grantor = test_agent_id();
        let grantee = test_agent_id();
        let future = Utc::now() + chrono::Duration::hours(1);
        let past = Utc::now() - chrono::Duration::hours(1);
        let perm_future = MemoryPermission::new(
            grantor, grantee, MemoryTier::Semantic, MemoryAccess::Read, MemoryScope::Group, Some(future),
        );
        let perm_past = MemoryPermission::new(
            grantor, grantee, MemoryTier::Semantic, MemoryAccess::Read, MemoryScope::Group, Some(past),
        );
        let perm_no_expiry = MemoryPermission::new(
            grantor, grantee, MemoryTier::Semantic, MemoryAccess::Read, MemoryScope::Group, None,
        );

        assert!(!perm_future.is_expired());
        assert!(perm_past.is_expired());
        assert!(!perm_no_expiry.is_expired());
    }

    #[test]
    fn test_permission_permits() {
        let g1 = test_agent_id();
        let g2 = test_agent_id();

        let read_only = MemoryPermission::new(g1, g2, MemoryTier::Semantic, MemoryAccess::Read, MemoryScope::Global, None);
        let write_only = MemoryPermission::new(g1, g2, MemoryTier::Semantic, MemoryAccess::Write, MemoryScope::Global, None);
        let read_write = MemoryPermission::new(g1, g2, MemoryTier::Semantic, MemoryAccess::ReadWrite, MemoryScope::Global, None);

        // Read-only grant
        assert!(read_only.permits(MemoryAccess::Read));
        assert!(!read_only.permits(MemoryAccess::Write));
        assert!(!read_only.permits(MemoryAccess::ReadWrite));

        // Write-only grant
        assert!(!write_only.permits(MemoryAccess::Read));
        assert!(write_only.permits(MemoryAccess::Write));
        assert!(!write_only.permits(MemoryAccess::ReadWrite));

        // ReadWrite grant
        assert!(read_write.permits(MemoryAccess::Read));
        assert!(read_write.permits(MemoryAccess::Write));
        assert!(read_write.permits(MemoryAccess::ReadWrite));
    }

    #[test]
    fn test_grant_table_basic() {
        let table = GrantTable::new();
        let grantor = test_agent_id();
        let grantee = test_agent_id();
        let other = test_agent_id();

        let perm = MemoryPermission::new(
            grantor, grantee, MemoryTier::Semantic, MemoryAccess::Read, MemoryScope::Group, None,        );

        // Grant
        table.grant(perm.clone());
        assert_eq!(table.grant_count(), 1);
        assert_eq!(table.list_grants_for(grantee).len(), 1);
        assert!(table.grantees_of(grantor).contains(&grantee));

        // Revoke specific
        table.revoke(grantee, grantor, MemoryTier::Semantic);
        assert_eq!(table.grant_count(), 0);
        assert!(table.list_grants_for(grantee).is_empty());

        // Revoke all for agent
        table.grant(perm.clone());
        table.revoke_all_for_agent(grantee);
        assert_eq!(table.grant_count(), 0);
    }

    #[test]
    fn test_grant_table_check_read() {
        let table = GrantTable::new();
        let owner = test_agent_id();
        let reader = test_agent_id();
        let stranger = test_agent_id();

        // Owner can always read their own memory
        assert!(table.check_read(owner, owner, MemoryTier::Semantic, MemoryScope::Private));
        assert!(table.check_read(owner, owner, MemoryTier::Semantic, MemoryScope::Group));
        assert!(table.check_read(owner, owner, MemoryTier::Semantic, MemoryScope::Global));

        // Global scope: anyone can read
        assert!(table.check_read(reader, owner, MemoryTier::Semantic, MemoryScope::Global));
        assert!(table.check_read(stranger, owner, MemoryTier::Semantic, MemoryScope::Global));

        // Private scope: only owner can read
        assert!(!table.check_read(reader, owner, MemoryTier::Semantic, MemoryScope::Private));

        // Group scope: requires grant
        assert!(!table.check_read(reader, owner, MemoryTier::Semantic, MemoryScope::Group));

        // Add explicit grant
        let perm = MemoryPermission::new(
            owner, reader, MemoryTier::Semantic, MemoryAccess::Read, MemoryScope::Group, None,
        );
        table.grant(perm);
        assert!(table.check_read(reader, owner, MemoryTier::Semantic, MemoryScope::Group));
    }

    #[test]    fn test_grant_table_check_write() {
        let table = GrantTable::new();
        let owner = test_agent_id();
        let writer = test_agent_id();

        // Owner can always write their own memory
        assert!(table.check_write(owner, owner, MemoryTier::Semantic, MemoryScope::Private));

        // Private scope: only owner can write
        assert!(!table.check_write(writer, owner, MemoryTier::Semantic, MemoryScope::Private));

        // Group/Global scope: requires explicit write grant
        assert!(!table.check_write(writer, owner, MemoryTier::Semantic, MemoryScope::Group));
        assert!(!table.check_write(writer, owner, MemoryTier::Semantic, MemoryScope::Global));

        // Add write grant
        let perm = MemoryPermission::new(
            owner, writer, MemoryTier::Semantic, MemoryAccess::Write, MemoryScope::Global, None,
        );
        table.grant(perm);
        assert!(table.check_write(writer, owner, MemoryTier::Semantic, MemoryScope::Global));
    }

    #[test]
    fn test_group_table_membership() {
        let groups = GroupTable::new();
        let a = test_agent_id();
        let b = test_agent_id();
        let c = test_agent_id();

        // Initially no memberships
        assert!(!groups.in_same_group(a, b));
        assert!(groups.agent_groups(a).is_empty());

        // Add to same group
        groups.add_to_group(a, "team-alpha".to_string());
        groups.add_to_group(b, "team-alpha".to_string());
        groups.add_to_group(c, "team-beta".to_string());

        assert!(groups.in_same_group(a, b));
        assert!(!groups.in_same_group(a, c));
        assert!(!groups.in_same_group(b, c));

        assert_eq!(groups.agent_groups(a), vec!["team-alpha"]);
        assert_eq!(groups.group_members("team-alpha").len(), 2);

        // Remove from group
        groups.remove_from_group(a, "team-alpha");
        assert!(!groups.in_same_group(a, b));
        assert!(groups.agent_groups(a).is_empty());    }

    #[test]
    fn test_access_checker_read_logic() {
        let grants = Arc::new(GrantTable::new());
        let groups = Arc::new(GroupTable::new());
        let checker = MemoryAccessChecker::new(grants, groups);

        let owner = test_agent_id();
        let group_member = test_agent_id();
        let granted = test_agent_id();
        let stranger = test_agent_id();

        // Set up group membership
        groups.add_to_group(owner, "research".to_string());
        groups.add_to_group(group_member, "research".to_string());

        // Set up explicit grant
        let perm = MemoryPermission::new(
            owner, granted, MemoryTier::Semantic, MemoryAccess::Read, MemoryScope::Group, None,
        );
        checker.grant_table().grant(perm);

        // Test cases
        // 1. Owner reads own: always allowed
        assert!(checker.check_read(owner, owner, MemoryScope::Private, MemoryTier::Semantic).is_ok());

        // 2. Global scope: anyone can read
        assert!(checker.check_read(stranger, owner, MemoryScope::Global, MemoryTier::Semantic).is_ok());

        // 3. Private scope: only owner
        assert!(checker.check_read(group_member, owner, MemoryScope::Private, MemoryTier::Semantic).is_err());

        // 4. Group scope + same group: allowed
        assert!(checker.check_read(group_member, owner, MemoryScope::Group, MemoryTier::Semantic).is_ok());

        // 5. Group scope + explicit grant: allowed
        assert!(checker.check_read(granted, owner, MemoryScope::Group, MemoryTier::Semantic).is_ok());

        // 6. Group scope + no group + no grant: denied
        assert!(checker.check_read(stranger, owner, MemoryScope::Group, MemoryTier::Semantic).is_err());
    }

    #[test]
    fn test_access_checker_write_logic() {
        let grants = Arc::new(GrantTable::new());
        let groups = Arc::new(GroupTable::new());
        let checker = MemoryAccessChecker::new(grants, groups);

        let owner = test_agent_id();        let other = test_agent_id();

        // Same group doesn't grant write access
        groups.add_to_group(owner, "team".to_string());
        groups.add_to_group(other, "team".to_string());

        // L1/L2 always private for writes
        assert!(checker.check_write(other, owner, MemoryScope::Group, MemoryTier::Working).is_err());
        assert!(checker.check_write(other, owner, MemoryScope::Global, MemoryTier::Episodic).is_err());

        // L3/L4: group membership alone doesn't grant write
        assert!(checker.check_write(other, owner, MemoryScope::Group, MemoryTier::Semantic).is_err());
        assert!(checker.check_write(other, owner, MemoryScope::Global, MemoryTier::Semantic).is_err());

        // Explicit write grant required
        let perm = MemoryPermission::new(
            owner, other, MemoryTier::Semantic, MemoryAccess::Write, MemoryScope::Global, None,
        );
        checker.grant_table().grant(perm);
        assert!(checker.check_write(other, owner, MemoryScope::Global, MemoryTier::Semantic).is_ok());
    }

    #[test]
    fn test_prune_expired() {
        let table = GrantTable::new();
        let g1 = test_agent_id();
        let g2 = test_agent_id();
        let g3 = test_agent_id();

        let expired = MemoryPermission::new(
            g1, g2, MemoryTier::Semantic, MemoryAccess::Read, MemoryScope::Group,
            Some(Utc::now() - chrono::Duration::seconds(1)),
        );
        let valid = MemoryPermission::new(
            g1, g3, MemoryTier::Semantic, MemoryAccess::Read, MemoryScope::Group,
            Some(Utc::now() + chrono::Duration::hours(1)),
        );

        table.grant(expired);
        table.grant(valid);

        assert_eq!(table.grant_count(), 2);
        table.prune_expired();
        assert_eq!(table.grant_count(), 1);
        assert!(table.list_grants_for(g2).is_empty());
        assert_eq!(table.list_grants_for(g3).len(), 1);
    }

    #[test]
    fn test_revoke_all_for_agent() {        let table = GrantTable::new();
        let a = test_agent_id();
        let b = test_agent_id();
        let c = test_agent_id();

        // a grants to b and c
        table.grant(MemoryPermission::new(a, b, MemoryTier::Semantic, MemoryAccess::Read, MemoryScope::Global, None));
        table.grant(MemoryPermission::new(a, c, MemoryTier::Semantic, MemoryAccess::Read, MemoryScope::Global, None));
        // b grants to a
        table.grant(MemoryPermission::new(b, a, MemoryTier::Semantic, MemoryAccess::Read, MemoryScope::Global, None));

        assert_eq!(table.grant_count(), 3);

        // Revoke all for b: removes grants to b AND grants from b
        table.revoke_all_for_agent(b);

        // b should have no grants
        assert!(table.list_grants_for(b).is_empty());
        // a's grant to b should be gone
        assert!(!table.grantees_of(a).contains(&b));
        // b's grant to a should be gone
        assert!(!table.grantees_of(b).contains(&a));
        // a's grant to c should remain
        assert!(table.grantees_of(a).contains(&c));

        assert_eq!(table.grant_count(), 1);
    }
}
