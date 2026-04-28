// crates/nexus-obs/src/tui/widgets/mesh_tree.rs

use ratatui::{
    layout::Rect,
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Table, TableRow},
    Frame,
};

use nexus_proto::agent::{AgentKind, AgentMeta, AgentStatus};

use crate::tui::theme::NexusTheme;

/// Renders the agent list as a styled table with status indicators.
pub fn render_agent_tree(
    f: &mut Frame,
    area: Rect,
    agents: &[AgentMeta],
    theme: &NexusTheme,
) {
    // Prepare table headers
    let header = TableRow::new(vec![
        Span::styled("Status", Style::default().add_modifier(Modifier::BOLD)),
        Span::styled("Name", Style::default().add_modifier(Modifier::BOLD)),
        Span::styled("Kind", Style::default().add_modifier(Modifier::BOLD)),
        Span::styled("Priority", Style::default().add_modifier(Modifier::BOLD)),
        Span::styled("Duration", Style::default().add_modifier(Modifier::BOLD)),
    ])
    .style(Style::default().fg(theme.primary));

    // Prepare rows
    let rows: Vec<TableRow> = agents
        .iter()
        .map(|agent| {
            let (icon, status_style) = theme.status_icon(status_to_string(&agent.status));
            let duration = calculate_duration(&agent.status);

            TableRow::new(vec![
                Span::styled(icon, status_style),
                Span::raw(&agent.name),
                Span::raw(kind_to_string(&agent.kind)),
                Span::raw(priority_to_string(agent.priority)),
                Span::styled(duration, Style::default().fg(theme.muted)),
            ])
        })
        .collect();

    // Create table widget
    let table = Table::new(rows, [        Constraint::Length(3),  // Status icon
        Constraint::Min(20),    // Name
        Constraint::Length(15), // Kind
        Constraint::Length(10), // Priority
        Constraint::Length(10), // Duration
    ])
    .header(header)
    .block(
        Block::default()
            .title(" Running Agents ")
            .title_style(theme.title)
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme.border)),
    )
    .style(Style::default().fg(theme.text))
    .highlight_style(Style::default().add_modifier(Modifier::REVERSED));

    f.render_widget(table, area);
}

/// Converts an `AgentStatus` to a human-readable string for display.
fn status_to_string(status: &AgentStatus) -> String {
    match status {
        AgentStatus::Pending { .. } => "pending".into(),
        AgentStatus::Running { .. } => "running".into(),
        AgentStatus::Suspended { .. } => "suspended".into(),
        AgentStatus::Completed { success, .. } => {
            if *success { "completed" } else { "failed" }.into()
        }
        AgentStatus::Failed { .. } => "failed".into(),
        AgentStatus::Terminating => "terminating".into(),
    }
}

/// Converts an `AgentKind` to a display string.
fn kind_to_string(kind: &AgentKind) -> String {
    match kind {
        AgentKind::Research => "research".into(),
        AgentKind::Writing => "writing".into(),
        AgentKind::CodeReview => "code_review".into(),
        AgentKind::Analysis => "analysis".into(),
        AgentKind::Planning => "planning".into(),
        AgentKind::Custom(s) => format!("custom:{}", s),
    }
}

/// Converts an `AgentPriority` to a display string.
fn priority_to_string(priority: nexus_proto::agent::AgentPriority) -> String {
    match priority {
        nexus_proto::agent::AgentPriority::Critical => "critical".into(),        nexus_proto::agent::AgentPriority::High => "high".into(),
        nexus_proto::agent::AgentPriority::Normal => "normal".into(),
        nexus_proto::agent::AgentPriority::Low => "low".into(),
        nexus_proto::agent::AgentPriority::Background => "background".into(),
    }
}

/// Calculates a human-readable duration string from an agent status.
fn calculate_duration(status: &AgentStatus) -> String {
    use chrono::{DateTime, Utc};
    let now = Utc::now();

    let start_time = match status {
        AgentStatus::Running { started_at, .. } => Some(*started_at),
        AgentStatus::Completed { finished_at, .. } => Some(*finished_at),
        AgentStatus::Failed { failed_at, .. } => Some(*failed_at),
        AgentStatus::Suspended { suspended_at, .. } => Some(*suspended_at),
        _ => None,
    };

    if let Some(t) = start_time {
        let dur = now.signed_duration_since(t);
        if dur.num_hours() > 0 {
            format!("{}h {}m", dur.num_hours(), dur.num_minutes() % 60)
        } else if dur.num_minutes() > 0 {
            format!("{}m {}s", dur.num_minutes(), dur.num_seconds() % 60)
        } else {
            format!("{}s", dur.num_seconds())
        }
    } else {
        "–".into()
    }
}
