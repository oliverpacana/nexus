use std::collections::{HashMap, HashSet};

use petgraph::algo::{is_cyclic_directed, toposort};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use tracing::{debug, instrument};

use nexus_proto::workflow::{StepId, StepDefinition, StepKind, WorkflowDefinition};
use crate::error::FlowError;

/// Type alias for the underlying petgraph directed graph.
pub type WorkflowGraph = DiGraph<StepDefinition, ()>;

/// The runtime DAG structure for a workflow definition.
///
/// Built during workflow initialization, it provides efficient graph traversal,
/// cycle detection, topological sorting, and Mermaid diagram generation.
#[derive(Debug)]
pub struct WorkflowDag {
    /// The directed graph where nodes are `StepDefinition`s and edges represent execution order.
    pub graph: WorkflowGraph,

    /// Maps human-readable `StepId` to internal `petgraph::NodeIndex` for O(1) lookup.
    pub node_index: HashMap<StepId, NodeIndex>,

    /// Internal index of the workflow's entry point.
    pub entry: NodeIndex,
}

impl WorkflowDag {
    /// Constructs a `WorkflowDag` from a `WorkflowDefinition`.
    ///
    /// Validates the DAG structure, detects cycles, verifies reachability,
    /// and ensures at least one `End` step is reachable.
    #[instrument(skip(definition))]
    pub fn build(definition: &WorkflowDefinition) -> Result<Self, FlowError> {
        let mut graph = DiGraph::new();
        let mut node_index = HashMap::with_capacity(definition.steps.len());

        // 1. Add all steps as nodes
        for (id, step_def) in &definition.steps {
            let idx = graph.add_node(step_def.clone());
            node_index.insert(id.clone(), idx);
        }

        // 2. Verify entry step exists
        let entry_idx = node_index
            .get(&definition.entry_step)
            .ok_or_else(|| FlowError::InvalidWorkflow {
                reason: format!("entry step '{}' not found in steps", definition.entry_step),            })?
            .clone();

        // 3. Add edges from `step.next` connections
        for (id, step_def) in &definition.steps {
            let source_idx = node_index[id];
            for next_id in &step_def.next {
                let target_idx = node_index.get(next_id).ok_or_else(|| {
                    FlowError::InvalidWorkflow {
                        reason: format!(
                            "step '{}' references unknown next step '{}'",
                            id, next_id
                        ),
                    }
                })?;
                graph.add_edge(source_idx, *target_idx, ());
            }
        }

        // 4. Detect cycles
        if is_cyclic_directed(&graph) {
            return Err(FlowError::InvalidWorkflow {
                reason: "workflow contains a cycle; must be a directed acyclic graph (DAG)".into(),
            });
        }

        // 5. Verify every node is reachable from entry (BFS)
        let mut visited = HashSet::new();
        let mut stack = vec![entry_idx];
        visited.insert(entry_idx);

        while let Some(curr) = stack.pop() {
            for neighbor in graph.neighbors(curr) {
                if visited.insert(neighbor) {
                    stack.push(neighbor);
                }
            }
        }

        if visited.len() < graph.node_count() {
            let unreachable: Vec<_> = node_index
                .iter()
                .filter(|(_, idx)| !visited.contains(idx))
                .map(|(id, _)| id.as_str().to_string())
                .collect();
            return Err(FlowError::InvalidWorkflow {
                reason: format!("unreachable steps detected: {:?}", unreachable),
            });
        }
        // 6. Verify at least one End step is reachable
        let has_reachable_end = visited.iter().any(|&idx| {
            matches!(
                graph.node_weight(idx).map(|s| &s.kind),
                Some(StepKind::End { .. })
            )
        });

        if !has_reachable_end {
            return Err(FlowError::InvalidWorkflow {
                reason: "no End step is reachable from the entry step".into(),
            });
        }

        debug!(
            nodes = graph.node_count(),
            edges = graph.edge_count(),
            "workflow DAG built successfully"
        );

        Ok(Self {
            graph,
            node_index,
            entry: entry_idx,
        })
    }

    /// Retrieves a step definition by its `StepId`.
    pub fn get_step(&self, id: &StepId) -> Option<&StepDefinition> {
        self.node_index
            .get(id)
            .and_then(|idx| self.graph.node_weight(*idx))
    }

    /// Returns the entry step definition.
    pub fn entry_step(&self) -> &StepDefinition {
        self.graph.node_weight(self.entry).expect("entry index is valid")
    }

    /// Returns all immediate successor steps for a given step.
    pub fn successors(&self, id: &StepId) -> Vec<&StepDefinition> {
        self.node_index
            .get(id)
            .map(|idx| {
                self.graph
                    .neighbors(*idx)
                    .filter_map(|n| self.graph.node_weight(n))
                    .collect()
            })
            .unwrap_or_default()    }

    /// Returns `true` if the node has multiple incoming edges (parallel join point).
    pub fn is_parallel_join(&self, id: &StepId) -> bool {
        self.node_index
            .get(id)
            .map(|idx| {
                self.graph
                    .edges_directed(*idx, petgraph::Direction::Incoming)
                    .count()
                    > 1
            })
            .unwrap_or(false)
    }

    /// Returns a topological ordering of all steps.
    /// Returns an error if the graph is cyclic (should not happen after successful `build`).
    pub fn topological_order(&self) -> Result<Vec<StepId>, FlowError> {
        match toposort(&self.graph, None) {
            Ok(indices) => Ok(indices
                .into_iter()
                .filter_map(|idx| self.graph.node_weight(idx).map(|s| s.id.clone()))
                .collect()),
            Err(_) => Err(FlowError::InvalidWorkflow {
                reason: "cycle detected during topological sort".into(),
            }),
        }
    }

    /// Returns the total number of nodes in the DAG.
    pub fn nodes_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Returns the total number of edges in the DAG.
    pub fn edges_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Renders the DAG as a Mermaid flowchart diagram.
    /// Useful for debugging, documentation, and visualization tools.
    ///
    /// Node shapes:
    /// - `[Step]` : Agent, Tool, Transform, Wait
    /// - `{Step}` : Conditional
    /// - `((Step))` : Parallel
    /// - `([Step])` : End
    pub fn to_mermaid(&self) -> String {
        let mut out = String::from("flowchart TD\n");
        // Track which edges we've rendered to avoid duplicates if graph has them
        let mut rendered_edges = HashSet::new();

        for node_idx in self.graph.node_indices() {
            let step = self.graph.node_weight(node_idx).unwrap();
            let id_str = sanitize_id(&step.id);

            // Determine Mermaid node shape based on step kind
            let (open, close) = match &step.kind {
                StepKind::Agent { .. } | StepKind::Tool { .. } | StepKind::Transform { .. } | StepKind::Wait { .. } => ("[", "]"),
                StepKind::Conditional { .. } => ("{", "}"),
                StepKind::Parallel { .. } => ("((", "))"),
                StepKind::End { .. } => ("([", "])"),
            };

            out.push_str(&format!(
                "    {}{}{}{}\n",
                id_str,
                open,
                escape_label(step.id.as_str()),
                close
            ));

            // Render outgoing edges
            for edge in self.graph.edges_directed(node_idx, petgraph::Direction::Outgoing) {
                let target = edge.target();
                let target_step = self.graph.node_weight(target).unwrap();
                let target_id_str = sanitize_id(&target_step.id);

                let edge_key = (id_str.clone(), target_id_str.clone());
                if rendered_edges.contains(&edge_key) {
                    continue;
                }
                rendered_edges.insert(edge_key);

                // Add label for conditional branches if applicable
                let label = if let StepKind::Conditional { branches, .. } = &step.kind {
                    branches
                        .iter()
                        .find(|(_, tid)| *tid == target_step.id)
                        .map(|(k, _)| format!("|{}|", escape_label(k)))
                        .unwrap_or_default()
                } else {
                    String::new()
                };

                out.push_str(&format!("    {} --> {}{}\n", id_str, label, target_id_str));
            }
        }
        out
    }
}

/// Sanitizes a step ID to be valid in Mermaid syntax.
fn sanitize_id(id: &StepId) -> String {
    id.as_str().replace(|c: char| !c.is_alphanumeric(), "_")
}

/// Escapes characters that could break Mermaid syntax.
fn escape_label(s: &str) -> String {
    s.replace(['"', '\n', '\r', '\\'], "")
        .replace('>', "&gt;")
        .replace('<', "&lt;")
}
