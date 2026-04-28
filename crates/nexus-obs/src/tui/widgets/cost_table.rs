// crates/nexus-obs/src/tui/widgets/cost_table.rs

use ratatui::{
    layout::{Constraint, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Row, Table, TableState},
    Frame,
};

use nexus_router::cost::CostRecord;

use crate::tui::theme::NexusTheme;

/// Renders a table of recent cost records with a total row.
pub fn render_cost_table(
    f: &mut Frame,
    area: Rect,
    records: &[CostRecord],
    theme: &NexusTheme,
) {
    // Prepare table headers
    let header = Row::new(vec![
        Span::styled("Agent", Style::default().add_modifier(Modifier::BOLD)),
        Span::styled("Provider", Style::default().add_modifier(Modifier::BOLD)),
        Span::styled("Model", Style::default().add_modifier(Modifier::BOLD)),
        Span::styled("In", Style::default().add_modifier(Modifier::BOLD)),
        Span::styled("Out", Style::default().add_modifier(Modifier::BOLD)),
        Span::styled("Cost ($)", Style::default().add_modifier(Modifier::BOLD)),
    ])
    .style(Style::default().fg(theme.primary));

    // Prepare rows
    let mut total_cost = 0.0;
    let rows: Vec<Row> = records
        .iter()
        .map(|rec| {
            total_cost += rec.estimated_cost_usd;
            Row::new(vec![
                Span::raw(rec.agent_id.to_string()),
                Span::raw(rec.provider.to_string()),
                Span::raw(&rec.model),
                Span::styled(rec.input_tokens.to_string(), Style::default().fg(theme.muted)),
                Span::styled(rec.output_tokens.to_string(), Style::default().fg(theme.muted)),
                Span::styled(
                    format!("{:.6}", rec.estimated_cost_usd),
                    Style::default().fg(theme.accent),
                ),
            ])
        })
        .collect();

    // Add total row
    let total_row = Row::new(vec![
        Span::styled("TOTAL", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw(""),
        Span::raw(""),
        Span::raw(""),
        Span::raw(""),
        Span::styled(
            format!("{:.6}", total_cost),
            Style::default()
                .fg(theme.secondary)
                .add_modifier(Modifier::BOLD),
        ),
    ]);

    // Create table widget
    let table = Table::new(
        rows,
        [
            Constraint::Length(36), // Agent ID (UUID)
            Constraint::Length(12), // Provider
            Constraint::Min(20),    // Model
            Constraint::Length(8),  // Input tokens
            Constraint::Length(8),  // Output tokens
            Constraint::Length(10), // Cost
        ],
    )
    .header(header)
    .row(total_row)
    .block(
        Block::default()
            .title(" Recent Costs ")
            .title_style(theme.title)
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme.border)),
    )
    .style(Style::default().fg(theme.text))
    .highlight_style(Style::default().add_modifier(Modifier::REVERSED));

    f.render_widget(table, area);
}
